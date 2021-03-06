import os
import shutil
import uuid
from os.path import join, dirname, realpath, exists
import tensorflow as tf

oplib_name="atr"
oplib_suffix=".so"


def _load_oplib(lib_name):
  """
  Load TensorFlow operator library.
  """
  lib_path = join(dirname(realpath(__file__)), 'lib{0}{1}'.format(lib_name, oplib_suffix))
  assert exists(lib_path), '{0} not found'.format(lib_path)

  # duplicate library with a random new name so that
  # a running program will not be interrupted when the lib file is updated
  lib_copy_path = '/tmp/lib{0}_{1}{2}'.format(lib_name, str(uuid.uuid4())[:8], oplib_suffix)
  shutil.copyfile(lib_path, lib_copy_path)
  oplib = tf.load_op_library(lib_copy_path)
  return oplib

_oplib = _load_oplib(oplib_name)

# map C++ operators to python objects
string_split_utf8 = _oplib.string_split_utf8
