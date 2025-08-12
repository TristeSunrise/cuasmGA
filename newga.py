# newga.py —— GA：仅用“相邻交换的内存指令安全移动”做变异
from typing import List, Optional, Callable
from collections import Counter
import random
import time, csv, hashlib, statistics
import numpy as np

from sass_kernel import SassKernel
from sassgen import write_sass_file


from sample import Sample

# ========= 超参数 =========
POP_SIZE        = 10
MUTATION_RATE   = 1.0    # 只靠变异，建议 1.0
NUM_GENERATIONS = 2000
ELITE_SIZE      = 4


class Individual:
    def __init__(self, kernel_section: List[str]):
        self.sass = kernel_section[:]     # 深拷贝
        self.fitness: float = float('inf')


class GeneticAlgorithm:
    """
    仅进行“相邻交换”的安全变异（只作用于内存指令）。
    - 不做 crossover（或恒等交叉）
    - 初始化：基线 + 若干次安全相邻交换得到的近邻
    - 评估：先 correctness gate，再性能
    """
    def __init__(
        self,
        kernel_section: List[str],
        sasskernel: SassKernel,
        test_correctness: Callable,
        test_performance: Callable[[Individual], float],
    ):
        self.original_kernel_section = kernel_section[:]
        self.sasskernel = sasskernel
        self.test_correctness = test_correctness
        self.test_performance = test_performance
        self.history = []              # 每代记录一条
        self._t0 = time.time() 
        self.mut_attempts = 0   # 变异尝试次数
        self.mut_moves    = 0   # 做成一次合法交换的次数
        self.mut_valids   = 0   # 变异后可运行的次数
        self.history = []              # 每代记录一条
        self._t0 = time.time() 
        # 多重集守恒
        self.counter = Counter(kernel_section)
        # 安全移动器
    @staticmethod
    def _sig(sass_lines):
        return hashlib.sha1("\n".join(sass_lines).encode()).hexdigest()[:12]

    def _record_gen(self, gen_idx: int, population):
        fits = [ind.fitness for ind in population]
        best_ind = min(population, key=lambda x: x.fitness)
        rec = {
            "gen": gen_idx,
            "best_fitness": best_ind.fitness,
            "best_sig": self._sig(best_ind.sass),
            "mean_fitness": float(sum(fits) / len(fits)),
            "median_fitness": float(statistics.median(fits)),
            "std_fitness": float(statistics.pstdev(fits)) if len(fits) > 1 else 0.0,
            "mut_attempts": int(self.mut_attempts),
            "mut_moves": int(self.mut_moves),
            "mut_valids": int(self.mut_valids),
            "move_rate": float(self.mut_moves / self.mut_attempts) if self.mut_attempts else 0.0,
            "valid_rate": float(self.mut_valids / max(1, self.mut_moves)),
            "elapsed_sec": float(time.time() - self._t0),
        }
        self.history.append(rec)
        self.mut_attempts = self.mut_moves = self.mut_valids = 0

    def save_history(self, path: str):
        if not self.history:
            return
        keys = list(self.history[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in self.history:
                w.writerow(row)
    # ---------- 评估 ----------
    def evaluate_fitness(self, individual: Individual) -> float:
        try:
            updated_sass = self.sasskernel._update_kernel(individual.sass)
            ok = self.test_correctness(write_sass_file(updated_sass))
            if not ok:
                individual.fitness = float("inf")
                return individual.fitness

            f = self.test_performance(individual)
            individual.fitness = float(f) if f is not None else float("inf")
            return individual.fitness
        except Exception:
            individual.fitness = float("inf")
            return individual.fitness

    # ---------- 初始化：基线----------
    def initialize_population(self, original_kernel_section: List[str]) -> List[Individual]:
        population: List[Individual] = []

        # 基线个体
        base = Individual(original_kernel_section[:])
        base.fitness = self.evaluate_fitness(base)
        population.append(base)

        # 不足时用基线拷贝补齐
        while len(population) < POP_SIZE:
            dup = Individual(base.sass[:])
            dup.fitness = base.fitness
            population.append(dup)

        return population

    # ---------- crossover：恒等交叉----------
    def crossover(self, parent1: Individual, parent2: Individual):
        c1 = Individual(parent1.sass[:]); c1.fitness = parent1.fitness
        c2 = Individual(parent2.sass[:]); c2.fitness = parent2.fitness
        return c1, c2

    # ---------- mutate：一次“相邻安全交换” ----------
    def mutate(self, individual: Individual) -> Individual:
        if random.random() >= MUTATION_RATE:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        self.mut_attempts += 1

        # 1) 构造一个 Sample
        sass = individual.sass[:]  # 拷贝一份做就地交换
        sample = Sample(sass) 
        dims, total, mem_loc, max_src_len = sample.static_analysis()

        if dims == 0:
            # 没有任何 mem 指令候选
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        # 2) 构造 dummy 的 space 拿到 masks
        n_feat = 10 + 1 + 1 + 1 + max_src_len
        dummy_space = np.zeros((1, total, n_feat), dtype=np.float32)
        _, masks = sample.embedding(dummy_space, mem_loc, max_src_len)  # masks: [[up,down], ...]

        # 3) 构造“离散动作空间”的合法动作列表（完全照 RL 的编码）
        #    action = idx*2 + dir  (dir: 0=上移, 1=下移)；这里不包含 noop
        valid_actions = []
        for i, (up, down) in enumerate(masks):
            if up:   valid_actions.append(i * 2 + 0)  # 上移
            if down: valid_actions.append(i * 2 + 1)  # 下移

        if not valid_actions:
            # 没有任何合法动作，相当于 noop
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        # 4) 采样一个合法动作并执行（完全复用 Sample.apply 的语义）
        action = random.choice(valid_actions)
        index, direction = divmod(action, 2)  # dir==0 上移；dir==1 下移（和 Env 一致）
        before = sample.kernel_section[:]
        sample.apply(index, direction)

        # 5) 多重集守恒（只是保险，交换相邻行理论上不会变）
        after = sample.kernel_section
        if Counter(after) != Counter(before):
            # 理论上不应发生；保守回退
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        # 6) 写回个体并评估
        individual.sass = after
        self.mut_moves += 1
        individual.fitness = self.evaluate_fitness(individual)
        if individual.fitness != float("inf"):
            self.mut_valids += 1
        return individual

    # ---------- 选择 ----------
    def tournament_selection(self, population, k=4):
        contenders = random.sample(population, k)
        return min(contenders, key=lambda x: x.fitness)

    # ---------- 主流程 ----------
    def run_ga(self, original_kernel: List[str]) -> Individual:
        print("testing the correctness of the original kernel")
        origin = Individual(original_kernel[:])
        origin.fitness = self.evaluate_fitness(origin)
        print(f"original kernel fitness: {origin.fitness}")

        population = self.initialize_population(original_kernel)
        self._record_gen(0,population)
        for gen in range(NUM_GENERATIONS):
            if gen%5 == 0 and self.mut_attempts > 0:
                print(f"success rate : {self.mut_valids/self.mut_attempts}")
                print(f"move rate:{self.mut_moves/self.mut_attempts}")
            best = min(population, key=lambda x: x.fitness)
            print(f"GEN {gen} best fitness: {best.fitness}")

            population.sort(key=lambda x: x.fitness)
            next_gen = population[:ELITE_SIZE]

            while len(next_gen) < POP_SIZE:
                p1 = self.tournament_selection(population, k=4)
                p2 = self.tournament_selection(population, k=4)
                c1, c2 = self.crossover(p1, p2)  # 恒等交叉
                if c1 and c1.fitness != float("inf"):
                    next_gen.append(self.mutate(c1))
                if len(next_gen) < POP_SIZE and c2 and c2.fitness != float("inf"):
                    next_gen.append(self.mutate(c2))

            population = next_gen
            self._record_gen(gen, population)

        best = min(population, key=lambda x: x.fitness)
        print(f"Best fitness:{ best.fitness}")
        print(f"success rate : {self.mut_valids/self.mut_attempts}")
        print(f"move rate:{self.mut_moves/self.mut_attempts}")
        return best
