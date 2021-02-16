import numpy as np 

def get_layer_activations(w, Gs, conv_name = '32x32/Conv1'): 
    w=w.reshape((1,18,-1))
    output=[]
  
    for layer_name, layer_output, layer_trainables in Gs.list_layers():
        if 'dlatents_in' in layer_name:
            output.append(layer_output)   
        if conv_name==layer_name:
            conv_activations=layer_output.eval(feed_dict={output[0].name: w})
            break
    return conv_activations #output


def get_directions(model):
    dirs=[]
    for i in range(1,len(model.layers)):
        dirs.append(model.layers[i].get_weights())
    return dirs


def edit_w_directions(edit_dict, w_original, directions):
    w_edit = w_original.copy()
    edit_dirs = np.zeros((1024,1))
    for au in edit_dict:
        edit_dirs+= edit_dict[au]*directions[au+2][0]@directions[au+2][2]@directions[au+2][4]
    
    w_edit[3:8] = (w_original+ 1*((directions[0][0])@(directions[1][0])@edit_dirs).reshape((18,512)))[3:8]
    
    return w_edit

def transfer_to_original(w_original, w_edit, lmin = 3, lmax= 8):
    w_final = w_original[0].copy()
    w_final[lmin:lmax] = w_edit.reshape((18,512))[lmin:lmax]
    return w_final.reshape((1,18,512))

