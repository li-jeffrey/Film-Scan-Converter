#!/bin/bash

# Python interpreter to use
PYTHON={path to your Film Scan Converter VENV  python executable}

# directory where Film-Scan-Converter is located
FSC_DIR={path to Film Scan Converter root dir}

"${PYTHON}" "${FSC_DIR}/source/Film Scan Converter.pyw" "-d" "$1" "-o" "$1/converted" &
