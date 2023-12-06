import numpy as np
from matplotlib import pyplot as plt
import grading.helpers as helpers


################################################################################
# Saving student results
################################################################################
def initialize_res(scope):
    exercise_id = "sciper"
    sciper_number = helpers.resolve('sciper_number', scope)
    stud_grad = dict(sciper_number=sciper_number)
    helpers.register_answer(exercise_id, stud_grad, scope)



# ------------------------- Example part -------------------------


def save_remove_faulty_feature(scope):
    exercise_id = 'remove_faulty_feature'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_X = grading_data["gr_X"]

     # Apply student's function and register results
    student_res = func(gr_X)
    helpers.register_answer(exercise_id, student_res, scope)

# ------------------------- Linear Classification part -------------------------

def save_add_bias_feature(scope):
    exercise_id = 'add_bias_feature'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_X = grading_data["gr_X"]

     # Apply student's function and register results
    student_res = func(gr_X)
    helpers.register_answer(exercise_id, student_res, scope)

def save_perceptron_step(scope):
    exercise_id = 'perceptron_step'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_X_list, gr_Y_list, gr_w_list = grading_data["gr_X_list"],  grading_data["gr_Y_list"],  grading_data["gr_w_list"]
     # Apply student's function and register results
    student_res_list = []
    for i in range(len(gr_X_list)):
        student_res_list.append(func(gr_X_list[i], gr_Y_list[i], gr_w_list[i]))
    helpers.register_answer(exercise_id, student_res_list, scope)

def save_perceptron_predict(scope):
    exercise_id = 'perceptron_predict'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_X, gr_w = grading_data["gr_X"],  grading_data["gr_w"]
     # Apply student's function and register results
    student_res = func(gr_X, gr_w)
    helpers.register_answer(exercise_id, student_res, scope)

def save_get_margin(scope):
    exercise_id = 'get_margin'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_X, gr_y, gr_w = grading_data["gr_X"], grading_data["gr_y"], grading_data["gr_w"]
     # Apply student's function and register results
    student_res = func(gr_X, gr_y, gr_w)
    helpers.register_answer(exercise_id, student_res, scope)


# ------------------------- kNN part -------------------------

def save_L2_dist(scope):
    exercise_id = 'L2_dist'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_v1, gr_v2 = grading_data["gr_v1"], grading_data["gr_v2"]
    student_res = func(gr_v1, gr_v2, None)
    helpers.register_answer(exercise_id, student_res, scope)
    
def save_L1_dist(scope):
    exercise_id = 'L1_dist'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_v1, gr_v2 = grading_data["gr_v1"], grading_data["gr_v2"]
    student_res = func(gr_v1, gr_v2, None)
    helpers.register_answer(exercise_id, student_res, scope)


def save_kNN_one_example_with_distance_func(scope):
    exercise_id = 'kNN_one_example_with_distance_func'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    def dummy_dist(v1, v2, p):
        return p*((v1 - v2).sum(axis=1))
    gr_unlabeled_example, gr_data, gr_labels = grading_data["gr_unlabeled_example"], grading_data["gr_data"], grading_data["gr_labels"]

    student_res = func(gr_unlabeled_example, dummy_dist, gr_data, gr_labels, 5, 3)
    helpers.register_answer(exercise_id, student_res, scope)

def save_KFold_cross_validation_KNN_for_p(scope):
    exercise_id = 'KFold_cross_validation_KNN_for_p'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_X, gr_Y = grading_data["gr_X"], grading_data["gr_Y"]

    student_res = func(gr_X, gr_Y, 5, 3)
    helpers.register_answer(exercise_id, student_res, scope)

# ------------------------- K-Means part -------------------------

def save_find_labels(scope):
    exercise_id = 'find_labels'
    func = helpers.resolve(exercise_id, scope)
    grading_data = helpers.get_data(exercise_id)

    gr_X, gr_C = grading_data["gr_X"], grading_data["gr_C"]

     # Apply student's function and register results
    student_res = func(gr_X, gr_C)
    helpers.register_answer(exercise_id, student_res, scope)

def save_average_cluster_distance(scope):
    exercise_id = 'average_within_cluster_distance'
    func = helpers.resolve(exercise_id, scope)
    grad_data = helpers.get_data(exercise_id)

    gr_X, gr_C, gr_y = grad_data["gr_X"], grad_data["gr_X"], grad_data["gr_y"]

     # Apply student's function and register results
    student_res = func(gr_X, gr_C, gr_y)
    helpers.register_answer(exercise_id, student_res, scope)


# ------------------------- YYY part -------------------------



