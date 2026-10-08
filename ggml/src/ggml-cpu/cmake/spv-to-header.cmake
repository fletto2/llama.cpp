# Writes a SPIR-V binary as a C array: cmake -DSPV=<in.spv> -DHDR=<out.h> -DNAME=<symbol> -P spv-to-header.cmake
file(READ ${SPV} hex HEX)
string(LENGTH "${hex}" n)
math(EXPR len "${n} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," arr "${hex}")
file(WRITE ${HDR} "// generated from the co-processing shader, do not edit\n#pragma once\n#include <stddef.h>\nstatic const unsigned char ${NAME}[] = {${arr}};\nstatic const size_t ${NAME}_len = ${len};\n")
