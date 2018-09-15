import tensorflow as tf
demo = tf.load_op_library('./libatr.so')

test_strings = tf.constant(['张辉', 'xx', 'ss', '中国人民共和国'], dtype=tf.string)
split_orig = tf.string_split(test_strings, '')
splits = demo.string_split_utf8(test_strings)
values = splits.values
indices = splits.indices
dense_shape = splits.dense_shape

print(values)
print(indices)
print(dense_shape)

with tf.Session() as sess:
    values_val = [v.decode() for v in sess.run(values)]
    print(values_val)
    print(sess.run(indices))
    print(sess.run(dense_shape))

    print(sess.run(split_orig.values))
    print(sess.run(split_orig.dense_shape))
    print(sess.run(split_orig.indices))
