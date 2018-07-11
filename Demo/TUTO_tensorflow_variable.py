import os

os.environ["PATH"] += ";D:/CUDA/v8.0/bin;"  # path of your CUDA_Toolkit/bin, *.dll is here
import tensorflow as tf

"""2018-07-11 Yonv1943 tutorial_TensorFlow_variable"""
"""2018-07-11 init_part_of_var"""


def run():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # low the warning level

    con_1 = tf.constant(1.0)
    con_9 = tf.constant(9.0)
    var_1 = tf.get_variable(name="var_1", shape=[], dtype=tf.float32)  # not assigned, scalar, shape=[],
    var_9 = tf.Variable(initial_value=9.0, name='name_of_var_9')  # not assigned yet, NOTICE!

    print("\n\n%6s |ASSIGN VARIABLE" % '')
    node_assign_1 = tf.assign(var_1, con_1)  # variable should be assigned before used
    node_add_9 = tf.add(var_1, con_9)
    var_list = [v for v in tf.global_variables() if 'of_var_9' in v.name]  # variable should be assigned before run
    init_part_of_var = tf.variables_initializer(var_list=var_list)
    with tf.Session() as sess:
        print("%6s |sess.run(init)" % sess.run(init_part_of_var))
        print("%6s |node_assign_1" % sess.run(node_assign_1))
        print("%6s |node_add_9" % sess.run(node_add_9))

    print("\n\n%6s |INIT VARIABLES" % '')
    init = tf.global_variables_initializer()  # for var_rd1, var_rd2
    with tf.Session() as sess:
        print("%6s |sess.run(init)" % sess.run(init))
        result = sess.run([var_1, var_9])
        print("%6s |type: %s |print: %s" % ('result', type(result), result))

    print("\n\n%6s |FEED VARIABLES or PLACEHOLDER" % '')
    input_ph = tf.placeholder(dtype=tf.float32, shape=[])
    node_add_1 = tf.add(input_ph, var_1)
    # node_assign_add_1 = tf.assign_add(con_1, var_9)  # it is wrong, constant can not assign_add
    node_assign_add_1 = tf.assign_add(var_9, con_1)  # it is right, variable can assign_add
    node_mul_43_10 = tf.multiply(node_add_1, node_assign_add_1)
    with tf.Session() as sess:
        print("%6s |sess.run(init)" % sess.run(init))
        print("%6s |== 43 * 10 == (42+1 * (9+=1))" % sess.run(node_mul_43_10, feed_dict={input_ph: 42.0, var_1: 1.0}))

    print("\n\n%6s |ERRORS & EXCEPTIONS in TensorFlow" % '')
    print("%6s |ASSIGN VARIABLE" % 'eg.')
    try:
        node_add_9 = tf.add(var_1, con_9)
        with tf.Session() as sess:
            print("%6s |node_add_9 without assigning var" % sess.run(node_add_9))
    except tf.errors.FailedPreconditionError:
        print("%6s |except Error: %s" % ('fix', "tf.errors.FailedPreconditionError"))
        node_assign_1 = tf.assign(var_1, con_1)
        node_add_9 = tf.add(var_1, con_9)

        with tf.Session() as sess:
            print("%6s |node_assign_1" % sess.run(node_assign_1))
            print("%6s |node_add_9" % sess.run(node_add_9))
    except Exception as e:
        print("Error:", e)

    print("\n\n%6s |PRINT tf.Print" % '')
    print("%6s |FEED VARIABLES or PLACEHOLDER" % 'eg.')
    input_ph = tf.placeholder(dtype=tf.float32, shape=[])
    node_add_1 = tf.add(input_ph, var_1)
    node_assign_add_1 = tf.assign_add(var_9, con_1)  # it is right, variable can assign_add
    node_mul_43_10 = tf.multiply(node_add_1, node_assign_add_1)
    node_print = tf.Print(node_add_1, [input_ph, var_1])  # an TensorFlow Operation
    with tf.Session() as sess:
        print("%6s |sess.run(init)" % sess.run(init))
        result = sess.run([node_mul_43_10, node_print], feed_dict={input_ph: 42.0, var_1: 1.0})
        print("%6s |== 43 * 10 == (42+1 * (9+=1))" % result[0])
        print("%6s |tf.Print(node_add_1, [input_ph, var_1])" % result[1])


if __name__ == '__main__':
    run()
