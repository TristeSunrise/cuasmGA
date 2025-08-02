from copy import deepcopy


class SassKernel:
    def __init__(self, sass_lines:list[str], kernel_lines:list[str]):
        self.sass = sass_lines
        self.kernel_section = kernel_lines

        self.kernel_label = None

        self.kernel_start_line = 0

        self.startline = 0
        self.endline = 0

        self._locate_kernel()

    def _locate_kernel(self) -> None:
        for i, line in enumerate(self.kernel_section):
            if '.text.' in line:
                self.kernel_label = line
                self.kernel_start_line = i
                break
            assert self.kernel_start_line is not None, 'kernel_start_line not found'

        for i, line in enumerate(self.sass):
            if line == self.kernel_label:
                self.startline = i
                break
            assert self.startline is not None, 'startline not found'
    
        line = self.startline
        k_line = self.kernel_start_line
        while line < len(self.sass) and k_line < len(self.kernel_section):
            if self.sass[line] != self.kernel_section[k_line]:
                self.endline = line
                break
            line += 1
            k_line += 1
        if self.endline is None:
            assert self.sass[line] == self.kernel_section[k_line], 'kernel section end at last'
            self.endline = line

    def _get_kernel(self):
        return self.sass[self.startline:self.endline]
    
    def _update_kernel(self, kernel_lines):
        updated_sass = deepcopy(self.sass)
        updated_sass[self.startline:self.endline] = kernel_lines
        return updated_sass