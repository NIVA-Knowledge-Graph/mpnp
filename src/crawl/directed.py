

def simple_directed_crawl(kg, to_be_explored):
    global explored
    explored = set([])
    return __find_neighbors_directed(kg, to_be_explored)


def directed_crawl_with_backstep_crawl(kg, to_be_explored):
    global explored
    explored = set([])
    return __find_neighbors_one_way_recursive_back_step(kg, to_be_explored)


def directed_crawl_with_backstep_on_first_crawl(kg, to_be_explored):
    global explored
    explored = set([])
    directed = __find_neighbors_directed(kg, to_be_explored)
    explored = set([])
    around = __find_local_neighbors(kg, to_be_explored)
    return set(directed + around)


def __find_neighbors_directed(kg, to_be_explored):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]
    s = [s for s, p, o in connected_by_objects]
    results_when_searching_subjects = __find_neighbors_one_way_recursive(kg, set(s) - explored, 'subject')
    connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]
    o = [o for s, p, o in connected_by_subject]
    results_when_searching_objects = __find_neighbors_one_way_recursive(kg, set(o) - explored, 'object')
    return results_when_searching_objects + results_when_searching_subjects + connected_by_objects + connected_by_subject


def __find_local_neighbors(kg, to_be_explored):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]
    s = [s for s, p, o in connected_by_objects]
    results_when_searching_subjects = __find_neighbors_non_recursive(kg, set(s) - explored)
    connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]
    o = [o for s, p, o in connected_by_subject]
    results_when_searching_objects = __find_neighbors_non_recursive(kg, set(o) - explored)
    return results_when_searching_objects + results_when_searching_subjects + connected_by_objects + connected_by_subject


def __find_neighbors_non_recursive(kg, to_be_explored):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]

    connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]

    return connected_by_objects + connected_by_subject


def __find_neighbors_one_way_recursive(kg, to_be_explored, direction='none'):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    if direction == 'subject':
        connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]
        s = [s for s, p, o in connected_by_objects]
        connected_by_direction = __find_neighbors_one_way_recursive(kg, set(s) - explored, 'subject')
        return connected_by_objects + connected_by_direction
    if direction == 'object':
        connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]
        o = [o for s, p, o in connected_by_subject]
        connected_by_direction = __find_neighbors_one_way_recursive(kg, set(o) - explored, 'object')
        return connected_by_subject + connected_by_direction


def __find_neighbors_one_way_recursive_back_step(kg, to_be_explored, direction='none'):
    global explored
    if not to_be_explored:
        return []
    explored = explored | to_be_explored
    connected_by_objects = [(s, p, o) for s, p, o in kg if o in to_be_explored]
    connected_by_subject = [(s, p, o) for s, p, o in kg if s in to_be_explored]
    connected_by_subject_direction = []
    connected_by_object_direction = []
    if direction == 'subject' or direction == 'none':
        s = [s for s, p, o in connected_by_objects]
        connected_by_subject_direction = __find_neighbors_one_way_recursive_back_step(kg, set(s) - explored, 'subject')
    if direction == 'object' or direction == 'none':
        o = [o for s, p, o in connected_by_subject]
        connected_by_object_direction = __find_neighbors_one_way_recursive_back_step(kg, set(o) - explored, 'object')
    return connected_by_objects + connected_by_subject + connected_by_subject_direction + connected_by_object_direction