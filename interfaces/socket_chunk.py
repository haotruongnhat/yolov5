import pickle

def chunk_send(data, socket, addr):
    buf = 4096
    i = 0
    j = buf
    msg = pickle.dumps(data)
    packet_len = pickle.dumps(len(msg))
    socket.sendto(packet_len, addr)

    # Send pickle
    while(i<len(msg)):
        if j>(len(msg)-1):
            j=(len(msg))
            
        ###send pickle chunks
        socket.sendto(msg[i:j], addr)

        i += buf
        j += buf
        
def chunk_recv(socket):
    buf = 4096
    Dlen = int(pickle.loads(socket.recv(buf)))
    pkl_str = bytearray()
    for i in range(0,int(int(Dlen)/buf)+1):
        data = socket.recv(buf)
        pkl_str += data

    data = pickle.loads(pkl_str)
    
    return data