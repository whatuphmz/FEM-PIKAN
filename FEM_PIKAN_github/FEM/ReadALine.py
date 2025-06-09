def read_a_line(input_file, output_file):

    # !c=======================================================c
    # !c...input parameters:                                   c
    # !c    ifr:     the input file handle                     c
    # !c    ifw:     the output(.ECHO) file handle             c
    # !c...output parameter:                                   c
    # !c    linestr: one line proper input from the input file c
    # !c                                                       c
    # !c...NOTE:                                               c
    # !c    1. the linestr should be long enough,              c 
    # !c       e.g. the lenght > 80 in most cases              c 
    # !c    2. the comments in the input file will be          c 
    # !c       written to the output file, but the empty       c
    # !c       line will be ignored.                           c
    # !c=======================================================c
    # !c
    
    
    # !c===============================================c
    # !c   将test.inp中文档读取出来,在写出到test.echo中    c
    # !c          以查看读取输入参数有无错误               c
    # !c===============================================c
    # !c
    read_ok = False  # Flag to indicate a proper input line has been found

    while not read_ok:
        line_str = input_file.readline()
        if not line_str:  # End of file check
            break
        line_str = line_str.rstrip('\n')  # Remove trailing newline character
        line_str = line_str.replace(',', '')

        # Check each character in the line
        for char in line_str:
            if char in ['#', '!', 'c', '*']:  # Comment line check
                output_file.write(f" {line_str:80}\n")
                break  # Move to the next line
            elif char not in [' ', '\t']:  # Check for non-space/tab character
                read_ok = True  # This line is proper input
                break

    return line_str if read_ok else None  # Return the proper input line if found

# Define the function 'appext' to append file extensions
def appext(filename, ext):
    ext = ext.strip().lstrip('.')
    return f"{filename}.{ext}"

def checklim(nowval, limval, str):
    if nowval > limval:
        print(f"Exceed the limit of {str}")
        print(f"limval is: {limval}, nowval is: {nowval}")
        raise ValueError(f"Exceeded the limit for {str}.")