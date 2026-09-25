"""Rebuild Gensim 4.4 Word2Vec BLAS declarations: a valid -1 dot is not an exception.

Requires a C compiler, Cython 3.0.12, setuptools, wheel, and numpy.
Only alters the active Python environment; leaves upstream source files untouched.
"""
import importlib.machinery
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


def main():
    import gensim
    import numpy
    if importlib.metadata.version('gensim') != '4.4.0':
        raise RuntimeError('Patch is specifically for gensim 4.4.0')
    package=Path(gensim.__file__).parent
    build=Path(__file__).resolve().parents[1]/'assets/gensim-blas-fix'
    source=build/'gensim/models'
    source.mkdir(parents=True,exist_ok=True)
    for name in ['word2vec_inner.pyx','word2vec_inner.pxd']:
        shutil.copy2(package/'models'/name,source/name)
    pxd=source/'word2vec_inner.pxd'
    text=pxd.read_text()
    assert text.count('except -1 nogil')==2
    pxd.write_text(text.replace('except -1 nogil','noexcept nogil'))
    headers=list(package.rglob('voidptr.h'))
    if headers:
        shutil.copy2(headers[0],source/'voidptr.h')
    elif not (source/'voidptr.h').exists():
        urllib.request.urlretrieve('https://raw.githubusercontent.com/piskvorky/gensim/4.4.0/gensim/models/voidptr.h',source/'voidptr.h')
    (build/'setup.py').write_text('''from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
setup(name='gensim-blas-fix', ext_modules=cythonize([
    Extension('gensim.models.word2vec_inner', ['gensim/models/word2vec_inner.pyx'],
              include_dirs=[numpy.get_include(),'gensim/models'])], compiler_directives={'language_level':3}))
''')
    subprocess.run([sys.executable,'setup.py','build_ext','--build-lib','lib'],cwd=build,check=True)
    compiled=[p for p in (build/'lib/gensim/models').iterdir() if any(p.name.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES)]
    assert len(compiled)==1
    destination=package/'models'/compiled[0].name
    # Atomic replace: do not truncate a shared library mapped by a running process.
    temporary=destination.with_suffix(destination.suffix+'.new')
    shutil.copy2(compiled[0],temporary)
    temporary.replace(destination)
    (package/'models/word2vec_blas_fix.txt').write_text('Gensim 4.4.0: sdot_ptr and dsdot_ptr except -1 -> noexcept\n')
    print('Patched:',destination,'\nRestart Python before running Word2Vec.')


if __name__=='__main__': main()
