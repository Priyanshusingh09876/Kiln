// Block-granular prefix (radix) tree with LRU eviction — C++ core of the
// prefix-cache-aware router.
//
// The Python version in kiln/serving/router.py is the reference; this one is
// what the gateway uses when the extension is built, because at ~10k RPS the
// per-request match()/insert() over a few hundred blocks dominates the
// gateway's CPU time. Same semantics, same tests.
//
// Design: each node keys its children by a 64-bit hash of the token block
// (FNV-1a over the token ids) and stores the full block for verification, so a
// hash collision degrades to a miss, never a false hit. Eviction is LRU over
// *leaf* nodes via an intrusive doubly-linked list; interior nodes are evicted
// only once they become leaves, which matches how paged-KV caches free blocks.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace kiln {

using Block = std::vector<int64_t>;

static inline uint64_t hash_block(const int64_t* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    uint64_t v = static_cast<uint64_t>(p[i]);
    for (int b = 0; b < 8; ++b) {
      h ^= (v >> (8 * b)) & 0xffull;
      h *= 1099511628211ull;
    }
  }
  return h;
}

struct Node {
  Block block;                                             // tokens of this block (empty for root)
  Node* parent = nullptr;
  std::unordered_map<uint64_t, std::unique_ptr<Node>> children;
  uint64_t last_access = 0;
  // intrusive LRU list over leaves
  Node* lru_prev = nullptr;
  Node* lru_next = nullptr;
  bool in_lru = false;
};

class BlockRadixTree {
 public:
  BlockRadixTree(int block_size, int64_t capacity_blocks)
      : block_size_(block_size), capacity_(capacity_blocks), root_(new Node()) {}

  // Number of leading tokens already present (multiple of block_size).
  int64_t match(const std::vector<int64_t>& ids) const { return match_ptr(ids.data(), ids.size()); }
  int64_t match_ptr(const int64_t* ids, size_t n) const {
    const Node* node = root_.get();
    int64_t matched = 0;
    const size_t nblocks = n / block_size_;
    for (size_t b = 0; b < nblocks; ++b) {
      const int64_t* p = ids + b * block_size_;
      auto it = node->children.find(hash_block(p, block_size_));
      if (it == node->children.end() || !same(it->second->block, p)) break;
      node = it->second.get();
      matched += block_size_;
    }
    return matched;
  }

  // Insert; returns number of new blocks created. Touches LRU for every node on the path.
  int64_t insert(const std::vector<int64_t>& ids) { return insert_ptr(ids.data(), ids.size()); }
  int64_t insert_ptr(const int64_t* ids, size_t n) {
    Node* node = root_.get();
    int64_t added = 0;
    const size_t nblocks = n / block_size_;
    const uint64_t now = ++clock_;
    for (size_t b = 0; b < nblocks; ++b) {
      const int64_t* p = ids + b * block_size_;
      const uint64_t h = hash_block(p, block_size_);
      auto it = node->children.find(h);
      Node* next;
      if (it == node->children.end() || !same(it->second->block, p)) {
        auto fresh = std::make_unique<Node>();
        fresh->block.assign(p, p + block_size_);
        fresh->parent = node;
        next = fresh.get();
        if (it != node->children.end()) {  // hash collision: replace (rare)
          detach_subtree(it->second.get());
          it->second = std::move(fresh);
        } else {
          node->children.emplace(h, std::move(fresh));
        }
        ++added;
        ++size_;
        // parent is no longer a leaf
        if (node->in_lru) lru_remove(node);
      } else {
        next = it->second.get();
      }
      next->last_access = now;
      if (next->children.empty()) lru_touch(next);
      node = next;
    }
    evict();
    return added;
  }

  int64_t size_blocks() const { return size_; }
  int block_size() const { return block_size_; }
  int64_t capacity() const { return capacity_; }
  void set_capacity(int64_t c) { capacity_ = c; evict(); }

 private:
  bool same(const Block& blk, const int64_t* p) const {
    for (int i = 0; i < block_size_; ++i)
      if (blk[i] != p[i]) return false;
    return true;
  }

  // ---- LRU list (head = least recently used)
  void lru_remove(Node* n) {
    if (!n->in_lru) return;
    if (n->lru_prev) n->lru_prev->lru_next = n->lru_next; else lru_head_ = n->lru_next;
    if (n->lru_next) n->lru_next->lru_prev = n->lru_prev; else lru_tail_ = n->lru_prev;
    n->lru_prev = n->lru_next = nullptr;
    n->in_lru = false;
  }
  void lru_touch(Node* n) {
    lru_remove(n);
    n->lru_prev = lru_tail_;
    n->lru_next = nullptr;
    if (lru_tail_) lru_tail_->lru_next = n; else lru_head_ = n;
    lru_tail_ = n;
    n->in_lru = true;
  }

  void detach_subtree(Node* n) {
    // remove every node under n (inclusive) from the LRU and the size count
    std::vector<Node*> stack{n};
    while (!stack.empty()) {
      Node* cur = stack.back();
      stack.pop_back();
      lru_remove(cur);
      --size_;
      for (auto& kv : cur->children) stack.push_back(kv.second.get());
    }
  }

  void evict() {
    while (size_ > capacity_ && lru_head_) {
      Node* victim = lru_head_;
      lru_remove(victim);
      Node* parent = victim->parent;
      if (!parent) break;
      const uint64_t h = hash_block(victim->block.data(), block_size_);
      parent->children.erase(h);
      --size_;
      if (parent != root_.get() && parent->children.empty()) lru_touch(parent);
    }
  }

  int block_size_;
  int64_t capacity_;
  int64_t size_ = 0;
  uint64_t clock_ = 0;
  std::unique_ptr<Node> root_;
  Node* lru_head_ = nullptr;
  Node* lru_tail_ = nullptr;
};

}  // namespace kiln

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "kiln C++ core: block radix tree for prefix-cache-aware routing";
  py::class_<kiln::BlockRadixTree>(m, "BlockRadixTree")
      .def(py::init<int, int64_t>(), py::arg("block_size") = 16, py::arg("capacity_blocks") = 4096)
      .def("match", &kiln::BlockRadixTree::match, py::arg("ids"))
      .def("insert", &kiln::BlockRadixTree::insert, py::arg("ids"))
      .def("match_array",
           [](const kiln::BlockRadixTree& t, py::array_t<int64_t, py::array::c_style | py::array::forcecast> a) {
             return t.match_ptr(a.data(), static_cast<size_t>(a.size()));
           }, py::arg("ids"), "zero-copy match on an int64 numpy array")
      .def("insert_array",
           [](kiln::BlockRadixTree& t, py::array_t<int64_t, py::array::c_style | py::array::forcecast> a) {
             return t.insert_ptr(a.data(), static_cast<size_t>(a.size()));
           }, py::arg("ids"), "zero-copy insert from an int64 numpy array")
      .def_property_readonly("size_blocks", &kiln::BlockRadixTree::size_blocks)
      .def_property_readonly("block_size", &kiln::BlockRadixTree::block_size)
      .def_property("capacity", &kiln::BlockRadixTree::capacity, &kiln::BlockRadixTree::set_capacity);
}
