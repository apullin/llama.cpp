#include <cstdint>

// Align one tiled BF16 query tensor for chunked paged attention.
//
// src: [1,H,Sq,D] TILE BF16 in DRAM, 1 <= Sq <= 32, one padded query tile row
// out: [1,H,32|64,D] TILE BF16 in DRAM
//
// One worker owns one head. It first zeroes every output tile for that head,
// then copies each real source row into q_offset + row. TT tiles split a row
// into two 32-byte face rows: the second half begins 512 bytes after the first.

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);
    const uint32_t rows = get_arg_val<uint32_t>(3);
    const uint32_t q_offset = get_arg_val<uint32_t>(4);
    const uint32_t out_tile_rows = get_arg_val<uint32_t>(5);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t tile_bytes = get_tile_size(0);
    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t scratch = get_write_ptr(0);

    auto* zero = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for(uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
        zero[i] = 0;
    }

    const uint32_t out_head_base = head * out_tile_rows * Wt;
    for(uint32_t tr = 0; tr < out_tile_rows; ++tr) {
        for(uint32_t w = 0; w < Wt; ++w) {
            noc_async_write(scratch, out.get_noc_addr(out_head_base + tr * Wt + w), tile_bytes);
        }
    }
    noc_async_write_barrier();

    const uint32_t src_head_base = head * Wt;
    for(uint32_t row = 0; row < rows; ++row) {
        const uint32_t src_face = (row / 16) * 1024 + (row % 16) * 32;
        for(uint32_t w = 0; w < Wt; ++w) {
            noc_async_read(src.get_noc_addr(src_head_base + w, src_face), scratch + w * 64, 32);
            noc_async_read(src.get_noc_addr(src_head_base + w, src_face + 512), scratch + w * 64 + 32, 32);
        }
        noc_async_read_barrier();

        const uint32_t out_row = q_offset + row;
        const uint32_t out_tile_row = out_row / 32;
        const uint32_t out_row_in_tile = out_row % 32;
        const uint32_t out_face =
            (out_row_in_tile / 16) * 1024 + (out_row_in_tile % 16) * 32;
        for(uint32_t w = 0; w < Wt; ++w) {
            const uint32_t page = out_head_base + out_tile_row * Wt + w;
            noc_async_write(scratch + w * 64, out.get_noc_addr(page, out_face), 32);
            noc_async_write(scratch + w * 64 + 32, out.get_noc_addr(page, out_face + 512), 32);
        }
        noc_async_write_barrier();
    }
}
