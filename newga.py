from dataclasses import dataclass
from collections import defaultdict, Counter
import random
from typing import List, Dict, Set, Tuple, Optional, Callable
from collections import Counter
import random
from typing import Callable, Optional
from sass_kernel import SassKernel
from sassgen import write_sass_file


POP_SIZE = 10   #population size
MUTATION_RATE = 0.1
NUM_GENERATIONS = 10
ELITE_SIZE = 4
# --- 把文本序列 <-> 基线ID序列 的映射（处理重复指令文本） ---
def _make_catalog(baseline: List[str]) -> Dict[str, List[int]]:
    cat: Dict[str, List[int]] = {}
    for idx, line in enumerate(baseline):
        cat.setdefault(line, []).append(idx)
    return cat

def _to_ids(seq: List[str], baseline: List[str], catalog=None) -> List[int]:
    if catalog is None:
        catalog = _make_catalog(baseline)
    counters = defaultdict(int)
    ids = []
    for s in seq:
        k = counters[s]
        if s not in catalog or k >= len(catalog[s]):
            raise ValueError(f"发现与基线不一致的指令文本：{s!r}")
        ids.append(catalog[s][k])
        counters[s] += 1
    return ids

def _to_lines(ids: List[int], baseline: List[str]) -> List[str]:
    return [baseline[i] for i in ids]

# --- 拓扑合法性检查（保险） ---
def _is_topological(order: List[int], preds: Dict[int, Set[int]]) -> bool:
    pos = {v: i for i, v in enumerate(order)}
    for v, ps in preds.items():
        pv = pos[v]
        for u in ps:
            if pos[u] >= pv:
                return False
    return True

# --- 列表调度（就绪集合里按 key 选择），把 “随机键/优先级” 解码为拓扑序 ---
def _list_schedule_by_keys(keys: List[float], preds: Dict[int, Set[int]]) -> List[int]:
    N = len(keys)
    indeg = [0]*N
    succs = defaultdict(list)
    for v, ps in preds.items():
        indeg[v] = len(ps)
        for u in ps:
            succs[u].append(v)

    ready = {i for i in range(N) if indeg[i] == 0}
    taken = [False]*N
    order: List[int] = []

    while len(order) < N:
        if not ready:
            raise RuntimeError("DAG 断言失败：ready 为空。")
        # 选 key 最小的就绪点；平手用索引打破（稳定）
        v = min(ready, key=lambda i: (keys[i], i))
        order.append(v)
        taken[v] = True
        ready.remove(v)
        for w in succs[v]:
            indeg[w] -= 1
            if indeg[w] == 0 and not taken[w]:
                ready.add(w)
    return order

# --- PPX（先序保持）交叉：合并两个合法拓扑序，产出仍合法拓扑序 ---
def _ppx_ids(p1: List[int], p2: List[int], preds: Dict[int, Set[int]], rng=None) -> List[int]:
    if rng is None:
        rng = random.Random()
    N = len(p1)
    pos1 = {v:i for i,v in enumerate(p1)}
    pos2 = {v:i for i,v in enumerate(p2)}

    indeg = [0]*N
    succs = defaultdict(list)
    for v, ps in preds.items():
        indeg[v] = len(ps)
        for u in ps:
            succs[u].append(v)

    ready = {i for i in range(N) if indeg[i] == 0}
    taken = [False]*N
    child: List[int] = []
    i1 = i2 = 0

    def next_ready(seq, start):
        j = start
        while j < N:
            v = seq[j]
            if (not taken[v]) and (v in ready):
                return j, v
            j += 1
        return None

    while len(child) < N:
        c1 = next_ready(p1, i1)
        c2 = next_ready(p2, i2)

        if c1 and c2:
            _, a = c1; _, b = c2
            if a == b:
                chosen = a
            else:
                # 谁在“另一个父代”更靠前，就选谁；平手随机
                if pos2[a] < pos1[b]:
                    chosen = a
                elif pos1[b] < pos2[a]:
                    chosen = b
                else:
                    chosen = a if rng.random() < 0.5 else b
        elif c1:
            _, chosen = c1
        elif c2:
            _, chosen = c2
        else:
            # 都取不到时，从 ready 里选 “pos1+pos2 最小”的
            chosen = min(ready, key=lambda v: pos1[v] + pos2[v])

        child.append(chosen)
        taken[chosen] = True
        if chosen in ready:
            ready.remove(chosen)
        for w in succs[chosen]:
            indeg[w] -= 1
            if indeg[w] == 0 and not taken[w]:
                ready.add(w)
        while i1 < N and taken[p1[i1]]: i1 += 1
        while i2 < N and taken[p2[i2]]: i2 += 1

    assert _is_topological(child, preds)
    return child

# --- 用“随机键+轻微扰动”的安全变异（重新列表调度，始终合法） ---
def _mutate_ids_by_key_jitter(ids: List[int], preds: Dict[int, Set[int]], strength: float = 0.05, rng=None) -> List[int]:
    if rng is None:
        rng = random.Random()
    N = len(ids)
    # 以当前顺序当作基础键（位置即键），加入微小噪声
    base_keys = [0.0]*N
    for pos, v in enumerate(ids):
        base_keys[v] = float(pos)
    # 随机挑一些点加/减小扰动
    k = max(1, N // 50)  # 约 2% 的节点扰动；可调
    chosen = rng.sample(range(N), k)
    for v in chosen:
        base_keys[v] += rng.uniform(-strength, strength) * N
    # 重新解码为合法拓扑序
    return _list_schedule_by_keys(base_keys, preds)

# ======== 下面是你原 GA 的“最小改造版本” ========

class Individual:
    def __init__(self, kernel_section: List[str]):
        self.sass = kernel_section
        self.fitness: Optional[float] = None

class GeneticAlgorithm:
    original_kernel_section: Optional[list] = None

    def __init__(self, kernel_section: List[str],
                 test_correctness,
                 test_performance: Callable[[Individual], float],
                 preds: Dict[int, Set[int]]):
        # 基线 & DAG
        self.original_kernel_section = kernel_section
        self.baseline = kernel_section[:]            # 用作全集/映射基准
        self.preds = preds
        self.catalog = _make_catalog(self.baseline)

        self.counter = Counter(kernel_section)
        print(f"Existing duplicate？：{any(c>1 for c in self.counter.values())}")

        self.test_correctness = test_correctness
        self.test_performance = test_performance

        # 预缓存“基线ID顺序”，便于编码/解码
        self.baseline_ids = list(range(len(self.baseline)))

    def _to_ids(self, seq: List[str]) -> List[int]:
        return _to_ids(seq, self.baseline, self.catalog)

    def _to_lines(self, ids: List[int]) -> List[str]:
        return _to_lines(ids, self.baseline)

    def evaluate_fitness(self, individual: Individual) -> float:
        # 可选：先做 correctness gate（强烈建议）
        if not self.test_correctness(write_sass_file(individual.sass)):
            # 不通过的个体直接给个极差分（或丢弃）
            individual.fitness = float("inf")
            return individual.fitness
        fitness = self.test_performance(individual)
        individual.fitness = fitness
        return fitness

    # ---- 初始化：用 “随机键 + 列表调度” 采样拓扑序（而不是 random.shuffle）----
    def initialize_population(self, original_kernel_section: List[str]):
        population = []
        N = len(original_kernel_section)
        for _ in range(POP_SIZE):
            keys = [random.random() for _ in range(N)]
            ids = _list_schedule_by_keys(keys, self.preds)
            sass = self._to_lines(ids)
            assert Counter(sass) == self.counter
            ind = Individual(sass)
            ind.fitness = self.evaluate_fitness(ind)
            population.append(ind)
        return population

    # ---- 交叉：PPX，保证子代仍是 DAG 的拓扑序 ----
    def crossover(self, parent1: Individual, parent2: Individual):
        p1 = self._to_ids(parent1.sass)
        p2 = self._to_ids(parent2.sass)
        child_ids_1 = _ppx_ids(p1, p2, self.preds)
        child_ids_2 = _ppx_ids(p2, p1, self.preds)
        c1 = Individual(self._to_lines(child_ids_1))
        c2 = Individual(self._to_lines(child_ids_2))
        # 评估（含 correctness gate）
        c1.fitness = self.evaluate_fitness(c1)
        c2.fitness = self.evaluate_fitness(c2)
        return c1, c2

    # ---- 变异：对当前顺序做“键扰动→重调度”，始终合法 ----
    def mutate(self, individual: Individual) -> Individual:
        if random.random() < MUTATION_RATE:
            ids = self._to_ids(individual.sass)
            new_ids = _mutate_ids_by_key_jitter(ids, self.preds, strength=0.05)
            individual.sass = self._to_lines(new_ids)
            individual.fitness = self.evaluate_fitness(individual)
        return individual

    # ---- 你的 run_ga 逻辑基本不变，仅初始化已换成合法拓扑采样 ----
    def run_ga(self, originol_pure_kernel: List[str]):
        population = self.initialize_population(originol_pure_kernel)

        for gen in range(NUM_GENERATIONS):
            best = min(population, key=lambda x: x.fitness)
            print(f"GEN {gen} best fitness: {best.fitness}")

            population.sort(key=lambda x: x.fitness)
            next_gen = population[:ELITE_SIZE]

            while len(next_gen) < POP_SIZE:
                parent1 = self.tournament_selection(population, k=4)
                parent2 = self.tournament_selection(population, k=4)
                child1, child2 = self.crossover(parent1, parent2)
                if child1 and child1.fitness != float("inf"):
                    next_gen.append(self.mutate(child1))
                if len(next_gen) < POP_SIZE and child2 and child2.fitness != float("inf"):
                    next_gen.append(self.mutate(child2))
            population = next_gen

        best = min(population, key=lambda x: x.fitness)
        return best

    # 你原来的 tournament_selection 保留
    def tournament_selection(self, population, k=4):
        contenders = random.sample(population, k)
        return min(contenders, key=lambda x: x.fitness)
