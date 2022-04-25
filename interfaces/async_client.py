#!/usr/bin/python
import socket
import pickle
import json 


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

def exchange(client_port, server_port, im_path, conf_thres):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", int(client_port)))

        chunk_send(im_path, s, ("127.0.0.1", int(server_port)))

        result_dict = chunk_recv(s)
        ## Post-process
        processed_dict = {}
        processed_dict["boxes"] = []
        class_name = "thep"
        for bbox, score in zip(result_dict["boxes"], result_dict["scores"]):
            if score >= conf_thres:
                processed_dict["boxes"].append(
                    {
                        str(score) : [int(v) for v in bbox]
                    }
                )
        return json.dumps(processed_dict)

if __name__ == "__main__":
    im_path = "01042022_2.jpg"
    ip = "127.0.0.1"
    server_addr = 50000
    client_addr = 50002

    result_dict = exchange(client_addr, server_addr, im_path, 0.25)
    print(result_dict)