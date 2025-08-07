import pickle
import os
import tempfile
import time
from venv import logger
from CuAsm.CubinFile import CubinFile
from CuAsm.CuAsmParser import CuAsmParser

def extract_kernel_sass(pkl_path):
    with open(pkl_path, 'rb') as f:
        cubin_data = pickle.load(f)['cubin']

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.cubin') as temp_file:
        temp_file.write(cubin_data)
        temp_filename = temp_file.name

    try:
        cf = CubinFile(temp_filename)
        #todo : rename the var
        text_buffer_1, text_buffer_2 = cf.dump_sass()
        
        sass = text_buffer_1.getvalue().split('\n')
        kernel_section = text_buffer_2.getvalue().split('\n')
        # print(type(kernel_section))
        # print(kernel_section)
        # # 保存完整的 sass（全部 .cuasm 文本）
        # with open('full_sass_dump.txt', 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(sass))

        # # 保存 kernel_section（仅当前 kernel 的 .text 区域）
        # with open('kernel_section_dump.txt', 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(kernel_section))
        return sass, kernel_section  
    finally:
        os.unlink(temp_filename)

def extract_kernel_sass_from_bin(bin):
        # get initial cubin and asm (the initial have to file IO)
    with tempfile.NamedTemporaryFile(mode='wb', delete=True) as temp_file:
        cubin = bin.asm['cubin']
        temp_file.write(cubin)
        temp_filename = temp_file.name
        cf = CubinFile(temp_filename)
        #todo : rename the var
        text_buffer_1, text_buffer_2 = cf.dump_sass()
        sass = text_buffer_1.getvalue().split('\n')
        kernel_section = text_buffer_2.getvalue().split('\n')
        # print(type(kernel_section))
        # print(kernel_section)
        # # 保存完整的 sass（全部 .cuasm 文本）
        # with open('full_sass_dump.txt', 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(sass))

        # # 保存 kernel_section（仅当前 kernel 的 .text 区域）
        # with open('kernel_section_dump.txt', 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(kernel_section))
        return sass, kernel_section  



def write_sass_file(updated_sass):
    cap = CuAsmParser()
    assemble_ok = True
    cubin =None
    try:
        cap.parse_from_buffer(updated_sass)
        cubin = cap.dump_cubin()
        self.update_cubin(cubin)
    except Exception as e:
        print(f'Assemble failed: {e}')
        assemble_ok = False
    
    return cubin

def run_selection(cubin_dir_path):
    data = None
    if os.listdir(cubin_dir_path) is not None:
        for i, file in enumerate(os.listdir(cubin_dir_path)):
            if file == 'cache_config.pkl':
                continue
            if not file.endswith('.pkl'):
                continue
            with open(os.path.join(cubin_dir_path, file), 'rb') as f:
                data = pickle.load(f)
                return data['cubin']
        
        logger.info(f"No cache found in {cubin_dir_path}")
    
