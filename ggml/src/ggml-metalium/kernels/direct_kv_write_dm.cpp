#include <cstdint>

// Direct one-tile-row KV scatter for Qwen DFlash.
//
// src:     [1,1,M,1024] TILE BF16 in DRAM, 1 <= M <= 32
// cache:   [1,1,S,1024] TILE BF16 in DRAM, S tile-aligned
// indices: M UINT32 values in one ROW_MAJOR DRAM stick
//
// One worker handles one update row. A tiled logical row consists of two 32-byte
// face rows in each width tile. Copying those face rows verbatim preserves BF16
// bits and avoids transpose/slice/shard/typecast/update-cache programs.

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t index_addr = get_arg_val<uint32_t>(2);
    const uint32_t update = get_arg_val<uint32_t>(3);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t cache_rows = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto cache_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto index_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();

    const uint32_t tile_bytes = get_tile_size(0);
    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);
    const auto cache = TensorAccessor(cache_args, cache_addr, tile_bytes);
    const auto indices = TensorAccessor(index_args, index_addr);

    const uint32_t index_l1 = get_write_ptr(1);
    noc_async_read(indices.get_noc_addr(0, update * sizeof(uint32_t)), index_l1, sizeof(uint32_t));
    noc_async_read_barrier();
    const uint32_t cache_row = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_l1);
    if(cache_row == UINT32_MAX || cache_row >= cache_rows) {
        return;
    }

    const uint32_t src_row = update;
    const uint32_t src_face = (src_row / 16) * 1024 + (src_row % 16) * 32;
    const uint32_t row_l1 = get_write_ptr(0);
    for(uint32_t w = 0; w < Wt; ++w) {
        noc_async_read(src.get_noc_addr(w, src_face), row_l1 + w * 64, 32);
        noc_async_read(src.get_noc_addr(w, src_face + 512), row_l1 + w * 64 + 32, 32);
    }
    noc_async_read_barrier();

    const uint32_t cache_tile_row = cache_row / 32;
    const uint32_t cache_row_in_tile = cache_row % 32;
    const uint32_t cache_face =
        (cache_row_in_tile / 16) * 1024 + (cache_row_in_tile % 16) * 32;
    for(uint32_t w = 0; w < Wt; ++w) {
        const uint32_t page = cache_tile_row * Wt + w;
        noc_async_write(row_l1 + w * 64, cache.get_noc_addr(page, cache_face), 32);
        noc_async_write(row_l1 + w * 64 + 32, cache.get_noc_addr(page, cache_face + 512), 32);
    }
    noc_async_write_barrier();
}
