from sass_kernel import SassKernel
from sassgen import write_sass_file


class Individual:
    def __init__(self, kernel_section):
        self.sass = kernel_section  # List of instructions (strings)
        self.fitness = None

def evaluate_fitness(individual):
    cubin = write_sass_file(individual.sass)  # 生成完整 sass 文件
    run_benchmark(cubin)
def run_benchmark(cubin):
    return 