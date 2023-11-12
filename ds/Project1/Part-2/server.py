import logging
from xmlrpc.server import SimpleXMLRPCServer
from socketserver import ThreadingMixIn

import server_fns

class RPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass

## initialize server instance and register its functions
server_instance = RPCServer(('localhost', 3000), allow_none=True, logRequests=True)
server_instance.register_instance(server_fns.Server())

if __name__ == '__main__':
    ## Entry point
    try:
        logging.info('|----------- server ready -----------|')
        server_instance.serve_forever()
    except KeyboardInterrupt:
        logging.info('|----------- keyboard interrupt found -----------|')
