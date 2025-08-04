from copy import deepcopy


class SassKernel:
    def __init__(self, sass_lines:list[str], kernel_lines:list[str]):
        self.sass = sass_lines
        self.kernel_section = kernel_lines

        self.kernel_label = None

        self.kernel_start_line = 0

        self.startline = 0
        self.endline = 0
        print("SassKernel init called")
        self._locate_kernel()

    def _locate_kernel(self) -> None:
        print("_locate_kernel called" )
        for i, line in enumerate(self.kernel_section):
            if line.strip().startswith(".text."):
                self.kernel_label = line
                self.kernel_start_line = i
                print(f"kernrl start line found: {self.kernel_start_line}")
                break
            # else:print("kernrl start line not found")
        # assert self.kernel_start_line == 0, 'kernel_start_line not found'

        for i, line in enumerate(self.sass):
            if line == self.kernel_label:
                self.startline = i
                print(f"startline found: {self.startline}")
                break
            # else:print("startline not found")
        # assert self.startline ==0, 'startline not found'
    
        endline = self.startline
        k_line = self.kernel_start_line
        while endline < len(self.sass) and k_line < len(self.kernel_section):
            if self.sass[endline] != self.kernel_section[k_line]:
                self.endline = endline
                print(f"endline found:{self.endline}")
                break
            endline += 1
            k_line += 1
        if self.endline == 0:
            # assert self.sass[line] == self.kernel_section[k_line], 'kernel section end at last'
            self.endline = endline
            print(f"endline found:{self.endline}")
            
        # print("run _locate_kernel")

    def _get_kernel(self):
        return self.sass[self.startline:self.endline]
    
    def _update_kernel(self, kernel_lines):
        updated_sass = deepcopy(self.sass)
        updated_sass[self.startline:self.endline] = kernel_lines
        return updated_sass
        