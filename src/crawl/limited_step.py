

def two_step_crawl(kg, to_be_explored):
    pass
    global explored
    explored = set([])
    return __find_neighbors(kg, to_be_explored)


def one_step_crawl(kg, to_be_explored):
    pass
    global explored
    explored = set([])
    return __find_neighbors_non_recursive(kg, to_be_explored)


def __find_neighbors(kg, to_be_explored):
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

