import numpy as np
def PS(I, r):
  assert len(I.shape) == 3
  assert r>0
  r = int(r)
  O = np.zeros((I.shape[0]*r, I.shape[1]*r, int(I.shape[2]/(r*2))))
  for x in range(O.shape[0]):
    for y in range(O.shape[1]):
      for c in range(O.shape[2]):
        c += 1
        a = np.floor(x/r).astype("int")
        b = np.floor(y/r).astype("int")
        d = c*r*(y%r) + c*(x%r)
        print(a, b, d)
        O[x, y, c-1] = I[a, b, d]
  return O
def PS2(I, r):
  assert len(I.shape) == 3
  assert r>0
  r = int(r)
  O = np.zeros((int(I.shape[0]/(r*2)), I.shape[1]*r,I.shape[2]*r))
  for x in range(O.shape[1]):
    for y in range(O.shape[2]):
      for c in range(O.shape[0]):
        c += 1
        a = np.floor(x/r).astype("int")
        b = np.floor(y/r).astype("int")
        d = c*r*(y%r) + c*(x%r)
        print(a, b, d)
        O[c-1,x, y] = I[a, b, d]
  return O

def pixel_shuffle(input_tensor, r):
    b, c, h, w = input_tensor.shape
    assert c % (r * r) == 0, f"Input channels should be divisible by r * r (received {c})"
    
    oc = c // (r * r)
    oh = h * r
    ow = w * r

    output_tensor = np.reshape(input_tensor, (b, oc, r, r, h, w))
    output_tensor = np.transpose(output_tensor, (0, 1, 4, 2, 5, 3))
    output_tensor = np.reshape(output_tensor, (b, oc, oh, ow))

    return output_tensor

# Test the pixel_shuffle function
input_shape = (4, 5, 6)
r = 2

input_channels = input_shape[0] * (r * r)
input_tensor = np.random.random((1, input_channels, input_shape[1], input_shape[2]))

output_tensor = pixel_shuffle(input_tensor, r)
print("Input shape: ", input_tensor.shape)
print("Output shape: ", output_tensor.shape)
# I = np.random.rand(4, 4, 4)
# r = 2
# O = PS(I, r)
# print(I.shape)
# O2 = PS2(I, r)
# print(O.shape)
# print(O2.shape)