import random
from xmlrpc.server import SimpleXMLRPCServer

print("------ Server working ------")

list1, list2 = [None] * 100, [None] * 5

def add(inp1,inp2):
    """
    return addition of two numbers
    """
    return (int(inp1) + int(inp2))

def as_add(inp1,inp2):
    """
    asynchronous addition of two integers
    """
    addition = add(inp1, inp2)
    output = random.randrange(1, 100)
    list1[output] = addition
    return output

def sorting(inp1,inp2,inp3,inp4,inp5):
    """
    sort the 5 input integers
    """
    return sorted([inp1, inp2, inp3, inp4, inp5])

def as_sorting(inp1,inp2,inp3,inp4,inp5):
    """
    asynchronous sorting of 5 input integers
    """
    list2.extend(sorted([inp1,inp2,inp3,inp4,inp5]))
    return ("------ array received to server ------ ")

def ret_arr(val):
    """
    return sorted array back from server async
    """
    list3 = list2[5:10]
    for _ in range(0, 5):
        list2.pop()
    return list3

def ret_add(val):
    """
    return addition back from server async
    """
    return list1[val]


## define server instance with parameters
server = SimpleXMLRPCServer(('localhost', 3000), logRequests=True, allow_none=True)

## register functions for server to use
server.register_function(add,"add")
server.register_function(as_add,"as_add")
server.register_function(ret_add,"ret_add")
server.register_function(ret_arr,"ret_arr")
server.register_function(sorting,"sorting")
server.register_function(as_sorting,"as_sorting")

server.serve_forever()
