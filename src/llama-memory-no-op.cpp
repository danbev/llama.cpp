#include "llama-memory-no-op.h"

#include "llama-batch.h"

#include <stdexcept>

class llama_memory_no_op_context : public llama_memory_context_i {
public:
    explicit llama_memory_no_op_context(llama_memory_status status) : status(status) {}

    llama_memory_no_op_context(llama_memory_no_op * mem, std::vector<llama_ubatch> ubatches) :
        status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

    ~llama_memory_no_op_context() = default;

    bool next() override {
        if (++i_next >= ubatches.size()) {
            return false;
        }
        return true;
    }

    bool apply() override {
        if (status != LLAMA_MEMORY_STATUS_SUCCESS) {
            return false;
        }
        if (mem && i_next < ubatches.size()) {
            mem->commit_ubatch(ubatches[i_next]);
        }
        return true;
    }

    llama_memory_status get_status() const override { return status; }

    const llama_ubatch & get_ubatch() const override {
        return ubatches[i_next];
    }

private:
    const llama_memory_status status;

    llama_memory_no_op * mem = nullptr;
    size_t i_next = 0;
    std::vector<llama_ubatch> ubatches;
};

llama_memory_no_op::llama_memory_no_op(uint32_t n_seq_max) : n_seq_max(n_seq_max) {
    pos_max.fill(-1);
    pos_min.fill(-1);
}

llama_memory_context_ptr llama_memory_no_op::init_batch(
        llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);

    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    while (true) {
        llama_ubatch ubatch = balloc.split_simple(n_ubatch);
        if (ubatch.n_tokens == 0) {
            break;
        }
        ubatches.push_back(std::move(ubatch));
    }

    if (balloc.get_n_used() < balloc.get_n_tokens()) {
        return std::make_unique<llama_memory_no_op_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_memory_no_op_context>(this, std::move(ubatches));
}

llama_memory_context_ptr llama_memory_no_op::init_full() {
    return std::make_unique<llama_memory_no_op_context>(this, std::vector<llama_ubatch>{});
}

llama_memory_context_ptr llama_memory_no_op::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    return std::make_unique<llama_memory_no_op_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_no_op::get_can_shift() const {
    return false;
}

void llama_memory_no_op::commit_ubatch(const llama_ubatch & ubatch) {
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            if (seq_id < 0 || seq_id >= (int32_t) LLAMA_MAX_SEQ) {
                continue;
            }
            if (pos_min[seq_id] < 0 || p < pos_min[seq_id]) {
                pos_min[seq_id] = p;
            }
            if (p > pos_max[seq_id]) {
                pos_max[seq_id] = p;
            }
        }
    }
}

void llama_memory_no_op::clear(bool data) {
    GGML_UNUSED(data);
    pos_max.fill(-1);
    pos_min.fill(-1);
}

bool llama_memory_no_op::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (seq_id < 0 || seq_id >= (int32_t) LLAMA_MAX_SEQ) {
        return true;
    }
    if (p0 <= 0 && (p1 < 0 || p1 > pos_max[seq_id])) {
        pos_min[seq_id] = -1;
        pos_max[seq_id] = -1;
    }
    return true;
}

void llama_memory_no_op::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    if (seq_id_src < 0 || seq_id_src >= (int32_t) LLAMA_MAX_SEQ ||
        seq_id_dst < 0 || seq_id_dst >= (int32_t) LLAMA_MAX_SEQ) {
        return;
    }
    pos_min[seq_id_dst] = pos_min[seq_id_src];
    pos_max[seq_id_dst] = pos_max[seq_id_src];
}

void llama_memory_no_op::seq_keep(llama_seq_id seq_id) {
    for (int32_t s = 0; s < (int32_t) LLAMA_MAX_SEQ; ++s) {
        if (s != seq_id) {
            pos_min[s] = -1;
            pos_max[s] = -1;
        }
    }
}

void llama_memory_no_op::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    if (seq_id < 0 || seq_id >= (int32_t) LLAMA_MAX_SEQ) {
        return;
    }
    if (pos_max[seq_id] >= 0) {
        pos_max[seq_id] += shift;
    }
    if (pos_min[seq_id] >= 0) {
        pos_min[seq_id] = std::max<llama_pos>(0, pos_min[seq_id] + shift);
    }
}

void llama_memory_no_op::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(d);
}

llama_pos llama_memory_no_op::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || seq_id >= (int32_t) LLAMA_MAX_SEQ) {
        return -1;
    }
    return pos_min[seq_id];
}

llama_pos llama_memory_no_op::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || seq_id >= (int32_t) LLAMA_MAX_SEQ) {
        return -1;
    }
    return pos_max[seq_id];
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_no_op::memory_breakdown() const {
    return {};
}

void llama_memory_no_op::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
}

void llama_memory_no_op::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
}
