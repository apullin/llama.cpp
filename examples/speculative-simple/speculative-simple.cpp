#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    const bool dflash_alt_tree = std::getenv("LLAMA_SPEC_DFLASH_ALT_TREE") != nullptr;
    const bool full_draft_block = std::getenv("LLAMA_SPEC_FULL_DRAFT_BLOCK") != nullptr;
    const bool detail_timing = std::getenv("LLAMA_SPEC_DFLASH_TIMING") != nullptr;
    const auto timing_now = [&]() -> int64_t {
        return detail_timing ? ggml_time_us() : 0;
    };
    const int32_t dflash_alt_n_seq = dflash_alt_tree ?
            1 + std::max(0, params.speculative.draft.n_max) : 1;
    if (dflash_alt_tree) {
        const bool has_dflash = std::find(
                params.speculative.types.begin(),
                params.speculative.types.end(),
                COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != params.speculative.types.end();
        if (!has_dflash) {
            LOG_ERR("%s", "LLAMA_SPEC_DFLASH_ALT_TREE requires --spec-type draft-dflash\n");
            return 1;
        }
        if (params.n_parallel < dflash_alt_n_seq) {
            LOG_ERR("LLAMA_SPEC_DFLASH_ALT_TREE requires -np >= %d (got %d)\n",
                    dflash_alt_n_seq, params.n_parallel);
            return 1;
        }
        LOG_INF("DFlash alternate verifier tree enabled: top-2 leaves, max sequences = %d\n",
                dflash_alt_n_seq);
    }
    if (full_draft_block) {
        LOG_INF("full speculative blocks enabled at the generation limit\n");
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model (if any) - this also creates the MTP draft context when MTP speculation is enabled
    common_speculative_init_result_ptr spec_init;

    {
        common_params params_dft = common_base_params_to_speculative(params);
        if (dflash_alt_tree) {
            params_dft.n_parallel = 1;
        }

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    llama_context * ctx_dft = params.speculative.draft.ctx_dft;

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    if (dflash_alt_tree && (use_ckpt_tgt || use_ckpt_dft)) {
        LOG_ERR("%s", "DFlash alternate verifier tree requires contexts with partial sequence removal\n");
        return 1;
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    if (spec == nullptr) {
        LOG_ERR("%s", "failed to initialize speculative decoding\n");
        return 1;
    }

    // eval the prompt on the target and feed it to the speculative implementation(s)
    {
        llama_batch batch_prompt = llama_batch_init(inp.size(), 0, 1);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            common_batch_add(batch_prompt, inp[i], i, { seq_id }, false);
        }

        llama_decode(ctx_tgt, batch_prompt);

        if (!common_speculative_process(spec, batch_prompt)) {
            LOG_ERR("%s", "failed to process speculative prompt\n");
            return 1;
        }
    }

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, dflash_alt_n_seq);

    llama_tokens draft;

    std::vector<llama_tokens> draft_candidates;

    auto * mem_tgt = llama_get_memory(ctx_tgt);

    int64_t n_alt_rows = 0;
    int64_t n_alt_rescues = 0;

    int64_t t_draft_us          = 0;
    int64_t t_target_prepare_us = 0;
    int64_t t_target_decode_us  = 0;
    int64_t t_spec_process_us   = 0;
    int64_t t_sample_us         = 0;
    int64_t t_commit_us         = 0;
    int64_t n_draft_calls_timing = 0;
    int64_t n_draft_tokens_timing = 0;
    int64_t n_target_calls_timing = 0;
    int64_t n_target_rows_timing = 0;

    common_prompt_checkpoint ckpt;

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // determine the max draft that fits the remaining context and generation budget
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            if (params.n_predict >= 0 && !full_draft_block) {
                n_draft_max = std::min(n_draft_max, params.n_predict - n_predict - 1);
            }
            n_draft_max = std::max(n_draft_max, 0);

            // generate a new draft
            draft_candidates.clear();
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .pos0       = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
                /* .candidates = */ dflash_alt_tree ? &draft_candidates : nullptr,
                /* .n_candidates = */ dflash_alt_tree ? 2 : 0,
            };
            const auto t_draft_start = timing_now();
            common_speculative_draft(spec);
            t_draft_us += timing_now() - t_draft_start;
            ++n_draft_calls_timing;
            n_draft_tokens_timing += draft.size();

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            // reset the draft context to the checkpoint before verification
            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // Always have a token to evaluate from before - id_last. In alternate-tree
        // mode, every usable top-2 token gets its own one-row branch. The shared
        // main-path rows carry all descendant sequence IDs, so a 14-token block
        // occupies 15 + 14 = 29 verifier rows instead of 15 and still fits one
        // 32-row Metalium tile.
        const auto t_target_prepare_start = timing_now();
        common_batch_clear(batch_tgt);
        std::vector<int32_t> main_rows;
        std::vector<int32_t> alt_rows(draft.size(), -1);
        std::vector<llama_seq_id> alt_seqs(draft.size(), -1);
        llama_seq_id selected_seq = seq_id;

        if (dflash_alt_tree) {
            llama_seq_id next_seq = 1;
            for (size_t i = 0; i < draft.size() && next_seq < dflash_alt_n_seq; ++i) {
                if (i < draft_candidates.size() && draft_candidates[i].size() >= 2 &&
                    draft_candidates[i][0] == draft[i] && draft_candidates[i][1] != draft[i]) {
                    alt_seqs[i] = next_seq++;
                }
            }

            for (llama_seq_id s = 1; s < dflash_alt_n_seq; ++s) {
                llama_memory_seq_rm(mem_tgt, s, -1, -1);
            }
            for (llama_seq_id s = 1; s < next_seq; ++s) {
                llama_memory_seq_cp(mem_tgt, seq_id, s, -1, -1);
            }

            std::vector<llama_seq_id> all_seqs;
            all_seqs.reserve(next_seq);
            for (llama_seq_id s = 0; s < next_seq; ++s) {
                all_seqs.push_back(s);
            }
            main_rows.push_back(batch_tgt.n_tokens);
            common_batch_add(batch_tgt, id_last, n_past++, all_seqs, true);

            for (size_t i = 0; i < draft.size(); ++i) {
                std::vector<llama_seq_id> descendants = { seq_id };
                for (size_t j = i + 1; j < alt_seqs.size(); ++j) {
                    if (alt_seqs[j] >= 0) {
                        descendants.push_back(alt_seqs[j]);
                    }
                }
                main_rows.push_back(batch_tgt.n_tokens);
                common_batch_add(batch_tgt, draft[i], n_past + i, descendants, true);
            }

            for (size_t i = 0; i < alt_seqs.size(); ++i) {
                if (alt_seqs[i] < 0) {
                    continue;
                }
                alt_rows[i] = batch_tgt.n_tokens;
                common_batch_add(batch_tgt, draft_candidates[i][1], n_past + i, { alt_seqs[i] }, true);
                ++n_alt_rows;
            }
        } else {
            main_rows.push_back(batch_tgt.n_tokens);
            common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);

            for (size_t i = 0; i < draft.size(); ++i) {
                main_rows.push_back(batch_tgt.n_tokens);
                common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
            }
        }

        t_target_prepare_us += timing_now() - t_target_prepare_start;
        ++n_target_calls_timing;
        n_target_rows_timing += batch_tgt.n_tokens;
        //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());
        const auto t_target_decode_start = timing_now();
        llama_decode(ctx_tgt, batch_tgt);
        t_target_decode_us += timing_now() - t_target_decode_start;

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        const auto t_spec_process_start = timing_now();
        const bool spec_process_ok = dflash_alt_tree || common_speculative_process(spec, batch_tgt);
        t_spec_process_us += timing_now() - t_spec_process_start;
        if (!spec_process_ok) {
            LOG_ERR("%s", "failed to process speculative batch\n");
            break;
        }

        // only save the sampler sampler state if we use checkpoints
        const auto t_sample_start = timing_now();
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // save the size of the draft being verified
        const size_t n_draft = draft.size();

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        llama_tokens ids;
        if (dflash_alt_tree) {
            size_t i = 0;
            for (; i < draft.size(); ++i) {
                const llama_token id = common_sampler_sample(smpl.get(), ctx_tgt, main_rows[i]);
                common_sampler_accept(smpl.get(), id, true);
                ids.push_back(id);

                if (id == draft[i]) {
                    continue;
                }

                if (alt_rows[i] >= 0 && i < draft_candidates.size() &&
                    draft_candidates[i].size() >= 2 && id == draft_candidates[i][1]) {
                    selected_seq = alt_seqs[i];
                    const llama_token id_next = common_sampler_sample(smpl.get(), ctx_tgt, alt_rows[i]);
                    common_sampler_accept(smpl.get(), id_next, true);
                    ids.push_back(id_next);
                    ++n_alt_rescues;
                }
                break;
            }

            if (i == draft.size()) {
                const llama_token id = common_sampler_sample(smpl.get(), ctx_tgt, main_rows[i]);
                common_sampler_accept(smpl.get(), id, true);
                ids.push_back(id);
            }
        } else {
            ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        }

        t_sample_us += timing_now() - t_sample_start;
        const auto t_process_path_start = timing_now();
        const bool process_path_ok = !dflash_alt_tree ||
            common_speculative_process_path(spec, batch_tgt, selected_seq, seq_id);
        t_spec_process_us += timing_now() - t_process_path_start;
        if (!process_path_ok) {
            LOG_ERR("%s", "failed to process selected speculative path\n");
            break;
        }

        const auto t_commit_start = timing_now();
        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, n_draft);

            draft = std::move(ids);

            {
                ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            if (ctx_dft) {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            t_commit_us += timing_now() - t_commit_start;
            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            if (dflash_alt_tree) {
                if (selected_seq != seq_id) {
                    llama_memory_seq_rm(mem_tgt, seq_id, -1, -1);
                    llama_memory_seq_cp(mem_tgt, selected_seq, seq_id, -1, -1);
                }

                for (llama_seq_id s = 1; s < dflash_alt_n_seq; ++s) {
                    llama_memory_seq_rm(mem_tgt, s, -1, -1);
                }
            }

            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);

            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            }
        }

        t_commit_us += timing_now() - t_commit_start;
        if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);
    if (dflash_alt_tree) {
        LOG_INF("alt_rows  = %lld\n", (long long) n_alt_rows);
        LOG_INF("alt_rescue= %lld\n", (long long) n_alt_rescues);
    }

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    common_speculative_print_stats(spec);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    if (detail_timing) {
        const int64_t total_us = t_dec_end - t_dec_start;
        const int64_t accounted_us = t_draft_us + t_target_prepare_us + t_target_decode_us +
            t_spec_process_us + t_sample_us + t_commit_us;
        const int64_t residual_us = std::max<int64_t>(0, total_us - accounted_us);
        const auto pct = [total_us](int64_t value) {
            return total_us > 0 ? 100.0 * value / total_us : 0.0;
        };
        LOG_INF("\n");
        LOG_INF("dflash timing: draft_calls=%lld draft_tokens=%lld target_calls=%lld target_rows=%lld\n",
                (long long) n_draft_calls_timing, (long long) n_draft_tokens_timing,
                (long long) n_target_calls_timing, (long long) n_target_rows_timing);
        LOG_INF("dflash timing: draft generation = %8.3f ms (%5.1f%%)\n", t_draft_us / 1e3, pct(t_draft_us));
        LOG_INF("dflash timing: target prepare   = %8.3f ms (%5.1f%%)\n", t_target_prepare_us / 1e3, pct(t_target_prepare_us));
        LOG_INF("dflash timing: target decode    = %8.3f ms (%5.1f%%)\n", t_target_decode_us / 1e3, pct(t_target_decode_us));
        LOG_INF("dflash timing: spec processing  = %8.3f ms (%5.1f%%)\n", t_spec_process_us / 1e3, pct(t_spec_process_us));
        LOG_INF("dflash timing: sampling         = %8.3f ms (%5.1f%%)\n", t_sample_us / 1e3, pct(t_sample_us));
        LOG_INF("dflash timing: commit/cleanup   = %8.3f ms (%5.1f%%)\n", t_commit_us / 1e3, pct(t_commit_us));
        LOG_INF("dflash timing: residual         = %8.3f ms (%5.1f%%)\n", residual_us / 1e3, pct(residual_us));
    }

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
