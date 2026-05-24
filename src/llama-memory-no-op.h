#pragma once

#include "llama-batch.h"
#include "llama-memory.h"

#include <array>

//
// llama_memory_no_op
//
// A llama_memory_i for backends whose KV-cache live entirely outside of
// llama.cpp (like the Hailo NPU backend for example where the device manages
// the cache).
class llama_memory_no_op : public llama_memory_i {
public:
    explicit llama_memory_no_op(uint32_t n_seq_max);
    ~llama_memory_no_op() = default;

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;
    bool get_can_shift() const override;
    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;

    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;

    void seq_keep(llama_seq_id seq_id) override;

    void seq_add (llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    void commit_ubatch(const llama_ubatch & ubatch);

private:
    const uint32_t n_seq_max;

    std::array<llama_pos, LLAMA_MAX_SEQ> pos_max;
    std::array<llama_pos, LLAMA_MAX_SEQ> pos_min;
};
