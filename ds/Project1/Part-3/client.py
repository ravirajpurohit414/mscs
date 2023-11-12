import argparse
from xmlrpc.client import ServerProxy

server_proxy = ServerProxy('http://localhost:3000', verbose=False)

if __name__ == '__main__':

    ## initialize a argument parser instance
    arg_parser = argparse.ArgumentParser()

    ## define the arguments and comments
    arg_parser.add_argument("-ads", "--add", type=str, nargs=2, metavar=('inp1', 'inp2'))
    arg_parser.add_argument("-adas", "--add_as", type=str, nargs=2, metavar=('inp1', 'inp2'))
    arg_parser.add_argument("-ss", "--sortarr", type=int, nargs=5, metavar=('inp1','inp2','inp3','inp4','inp5'))
    arg_parser.add_argument("-sas", "--sort_arr", type=int, nargs=5, metavar=('inp1','inp2','inp3','inp4','inp5'))
    arg_parser.add_argument("-ack1", "--ackSever1", type=int, nargs=1, metavar=('inp1'))
    arg_parser.add_argument("-ack2", "--ackSever2", type=int, nargs=1, metavar=('inp1'))

    ## pass arguments to process
    parsed_args = arg_parser.parse_args()

    ## for addition
    if parsed_args.add is not None:
        inp1, inp2 = parsed_args.add[0], parsed_args.add[1]
        sum = server_proxy.add(inp1, inp2)
        print("----- sum of numbers "+ str(inp1) + " &  " + str(inp2) + " sync server ------ " + str(sum))

    ## for asynchronous addition
    elif parsed_args.add_as is not None:
        inp1 = parsed_args.add_as[0]
        inp2 = parsed_args.add_as[1]
        print("----- request sent with id number ------ "+ str(server_proxy.as_add(inp1, inp2)))

    ## for sorting array
    elif parsed_args.sortarr is not None:
        inp1,inp2,inp3,inp4,inp5 = parsed_args.sortarr[0],parsed_args.sortarr[1],parsed_args.sortarr[2],parsed_args.sortarr[3],parsed_args.sortarr[4]
        sorted_array = server_proxy.sorting(inp1, inp2, inp3, inp4, inp5)
        print("----- sorted array from sync server ----- "+str(sorted_array))

    ## for sorting array asynchronously
    elif parsed_args.sort_arr is not None:
        inp1 = parsed_args.sort_arr[0]
        inp2 = parsed_args.sort_arr[1]
        inp3 = parsed_args.sort_arr[2]
        inp4 = parsed_args.sort_arr[3]
        inp5 = parsed_args.sort_arr[4]

        output = str(server_proxy.as_sorting(inp1, inp2, inp3, inp4, inp5))
        print(output)

    ## for receiving first acknowledgement
    elif parsed_args.ackSever1 is not None:
        inp1 = parsed_args.ackSever1[0]
        output = server_proxy.ret_add(inp1)
        print("------ sum of numbers from async server ------ "+ str(output))

    ## for receiving acknowledgement2
    elif parsed_args.ackSever2 is not None:
        inp1 = parsed_args.ackSever2[0]
        output = server_proxy.ret_arr(inp1)
        print("------ sorted array from async server ------ "+ str(output))