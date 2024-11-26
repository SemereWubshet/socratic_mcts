
def gen_num(x:int, y:int) -> None:
    if y == 0: return None
    for index_x in range(x):
        print("I'm index X: ", index_x)
        print("I'm y: ", y)
        gen_num(index_x, y-1)
    return None

l = []
def gen_num2(x: int, y: int) -> None:
    if y == 0: return None
    for i in range(x):  # Iterate over the first range
        for j in range(x):
            print(f"{i}, {j}, {y}")
            l.append((i,y))
            gen_num2(x, y-1)
    return None

gen_num2(2, 2)
print(len(l))

def gen_num3(x: int, y: int) -> None:
    def helper(ix: int, max_x: int, max_y: int) -> None:
        if ix >= max_x:  # Base case: stop when `ix` exceeds range
            return
        for jy in range(max_y + 1):  # Iterate through all `y` values
            print(f"{ix}, {jy}")  # Print the current pair
        helper(ix + 1, max_x, max_y)  # Recurse for the next value of `x`

    helper(0, x, y)  # Start recursion from `x = 0`



# gen_num3(2, 2)


"""
def gen_tree(student_node:StudentNode, tree_width:int, tree_depth:int) -> None:
    if tree_depth == 0:
        return None
    for width in range(tree_width):
        teacher_node = student_node.query()
        # Score teacher reply and save it here
        for width in range(tree_width):
            student_node = teacher_node.query()
            gen_tree(student_node, tree_width, tree_depth-1)
"""