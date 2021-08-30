### Probando la hoja de trucos de Numpy y su investigación de uso y campo de aplicación

#### Utilizando la hoja de trucos de Numpy, pruebe cada uno de los códigos y muestre los resultados de cada sección

#### - Hoja de trucos de Numpy


#### Creando Arrays y Initial Placheholders


```python
 import numpy as np
a = np.array([1,2,3])
print(a)
 b = np.array([(1.5,2,3), (4,5,6)], dtype = float)
print(b)
c = np.array([[(1.5,2,3), (4,5,6)], [(3,2,1), (4,5,6)]],dtype = float)
print(c)
```

    [1 2 3]
    [[1.5 2.  3. ]
     [4.  5.  6. ]]
    [[[1.5 2.  3. ]
      [4.  5.  6. ]]
    
     [[3.  2.  1. ]
      [4.  5.  6. ]]]
    


```python
#Create an array of zeros
np.zeros((3,4)) 
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
#Create an array of ones
np.ones((2,3,4),dtype=np.int16) 
```




    array([[[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]],
    
           [[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]], dtype=int16)




```python
#Create an array of evenly
d = np.arange(10,25,5) 
print(d)
```

    [10 15 20]
    


```python
#Create an array of evenly
# spaced values (number of samples)
np.linspace(0,2,9) 
```




    array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ])




```python
#Create a constant array  
e = np.full((2,2),7) 
print(e)
```

    [[7 7]
     [7 7]]
    


```python
#Create a 2X2 identity matrix
f = np.eye(2)
print(f)
```

    [[1. 0.]
     [0. 1.]]
    


```python
#Create an array with random values
np.random.random((2,2))
```




    array([[0.3909913 , 0.17469623],
           [0.97482956, 0.75841279]])




```python
#Create an empty array
np.empty((3,2)) 
```




    array([[0., 0.],
           [0., 0.],
           [0., 0.]])



#### Guardar y cargar en disco


```python
np.save('my_array', a)
np.savez('array.npz', a, b)
np.load('my_array.npy')
```




    array([1, 2, 3])



#### Guardar y cargar archivos de texto


```python
 np.loadtxt("myfile.txt")
np.genfromtxt("my_file.csv", delimiter=',')
 np.savetxt("myarray.txt", a, delimiter=" ")
```

#### Tipos de datos


```python
 np.int64  #Signed 64-bit integer types
np.float32 #Standard double-precision floating point
 np.complex #Complex numbers represented by 128 floats
np.bool #Boolean type storing TRUE and FALSE values
 np.object #Python object type
np.string_ #Fixed-length string type
 np.unicode_ #Fixed-length unicode type
```

#### Inspección de su matriz


```python
# Array dimensions
a.shape 
```




    (3,)




```python
#Length of array 
len(a)
```




    3




```python
#Number of array dimensions
b.ndim
```




    2




```python
#Number of array elements
e.size
```




    4




```python
#Data type of array elements
b.dtype
```




    dtype('float64')




```python
# Name of data type 
b.dtype.name 
```




    'float64'




```python
#Convert an array to a different type
b.astype(int) 
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
# Asking For Help
np.info(np.ndarray.dtype)
```

    Data-type of the array's elements.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    d : numpy dtype object
    
    See Also
    --------
    numpy.dtype
    
    Examples
    --------
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.dtype
    dtype('int32')
    >>> type(x.dtype)
    <type 'numpy.dtype'>
    

### Matemáticas de matriz
#### Operaciones aritmeticas


```python
#Subtraction
g = a - b 
print(g)
```

    [[-0.5  0.   0. ]
     [-3.  -3.  -3. ]]
    


```python
### Subtraction Addition
np.subtract(a,b)
b+a
```




    array([[2.5, 4. , 6. ],
           [5. , 7. , 9. ]])




```python
## Addition Division 
np.add(b,a) 
a / b 
```




    array([[0.66666667, 1.        , 1.        ],
           [0.25      , 0.4       , 0.5       ]])




```python
#Division Multiplication
np.divide(a,b) 
a * b 
```




    array([[ 1.5,  4. ,  9. ],
           [ 4. , 10. , 18. ]])




```python
#Multiplication
np.multiply(a,b)

```




    array([[ 1.5,  4. ,  9. ],
           [ 4. , 10. , 18. ]])




```python
#Exponentiation
np.exp(b)
```




    array([[  4.48168907,   7.3890561 ,  20.08553692],
           [ 54.59815003, 148.4131591 , 403.42879349]])




```python
#Square root
np.sqrt(b)
```




    array([[1.22474487, 1.41421356, 1.73205081],
           [2.        , 2.23606798, 2.44948974]])




```python
#Print sines of an array
np.sin(a)
```




    array([0.84147098, 0.90929743, 0.14112001])




```python
#Element-wise cosine 
np.cos(b)
```




    array([[ 0.0707372 , -0.41614684, -0.9899925 ],
           [-0.65364362,  0.28366219,  0.96017029]])




```python
#Element-wise natural logarithm
np.log(a)
```




    array([0.        , 0.69314718, 1.09861229])



#### Comparación



```python
#Element-wise comparison
a == b
```




    array([[False,  True,  True],
           [False, False, False]])




```python
#Element-wise comparison
a < 2 
```




    array([ True, False, False])




```python
#Array-wise comparison
np.array_equal(a, b) 
```




    False



#### Funciones agregadas


```python
#Array-wise sum
a.sum() 
```




    6




```python
#Array-wise minimum value
a.min()
```




    1




```python
#Maximum value of an array row
b.max(axis=0)
```




    array([4., 5., 6.])




```python
#Cumulative sum of the elements
b.cumsum(axis=1) 
```




    array([[ 1.5,  3.5,  6.5],
           [ 4. ,  9. , 15. ]])




```python
#Mean
a.mean() 
```




    2.0




```python
#Median
b.median()
```


```python
#Correlation coefficient
 a.corrcoef()
```


```python
#Standard deviation
 np.std(b) 
```

#### Copiar matrices



```python
#Create a view of the array with the same data
h = a.view() 
print(h)
```

    [1 2 3]
    


```python
#Create a copy of the array
np.copy(a)
```




    array([1, 2, 3])




```python
#Create a deep copy of the array
h = a.copy()
print(h)
```

    [1 2 3]
    

#### Ordenación de matrices


```python
#Sort an array
a.sort() 
print(a)
```

    [1 2 3]
    


```python
#Sort the elements of an array's axis
c.sort(axis=0) 
print(a)
```

    [1 2 3]
    

### Subconjunto, rebanado, indexación

#### Subconjunto


```python
#Select the element at the 2nd index
a[2] 
```




    3




```python
#Select the element at row 1 column 2
b[1,2]
```




    6.0



#### Rebanar


```python
#Select items at index 0 and 1
a[0:2]
```




    array([1, 2])




```python
#Select items at rows 0 and 1 in column 1
b[0:2,1]
```




    array([2., 5.])




```python
#Select all items at row 0
b[:1] 

```




    array([[1.5, 2. , 3. ]])




```python
#Reversed array a 
a[ : :-1]
```




    array([3, 2, 1])



#### Indexación booleana


```python
#Select elements from a less than 2
a[a<2] 
```




    array([1])



#### Indexación elegante


```python
#Select elements (1,0),(0,1),(1,2) and (0,0)
b[[1, 0, 1, 0],[0, 1, 2, 0]]
```




    array([4. , 2. , 6. , 1.5])




```python
#Select a subset of the matrix’s rows and columns 
b[[1, 0, 1, 0]][:,[0,1,2,0]]
```




    array([[4. , 5. , 6. , 4. ],
           [1.5, 2. , 3. , 1.5],
           [4. , 5. , 6. , 4. ],
           [1.5, 2. , 3. , 1.5]])



### Manipulación de matrices

#### Transposición de matriz


```python
#Permute array dimensions
 i = np.transpose(b)
#Permute array dimensions
i.T
```

#### Cambiar la forma de la matriz



```python
#Flatten the array
b.ravel() 
```




    array([1.5, 2. , 3. , 4. , 5. , 6. ])




```python
#Reshape, but don’t change data
 g.reshape(3,-2) 
    
```

#### Agregar / eliminar elementos


```python
# Return a new array with shape (2,6)
h.resize(2,6)
print(h)
```

    [[1 2 3 0 0 0]
     [0 0 0 0 0 0]]
    


```python
#Append items to an array
np.append(h,g) 
```




    array([ 1. ,  2. ,  3. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
            0. , -0.5,  0. ,  0. , -3. , -3. , -3. ])




```python
#Insert items in an array
np.insert(a, 1, 5) 
```




    array([1, 5, 2, 3])




```python
#Delete items from an array
np.delete(a,[1]) 
```




    array([1, 3])



#### Combinando matrices



```python
#Concatenate arrays
np.concatenate((a,d),axis=0) 
```




    array([ 1,  2,  3, 10, 15, 20])




```python
#Stack arrays vertically (row-wise)
np.vstack((a,b)) 
```




    array([[1. , 2. , 3. ],
           [1.5, 2. , 3. ],
           [4. , 5. , 6. ]])




```python
#Stack arrays vertically (row-wise)
np.r_[e,f] 
```




    array([[7., 7.],
           [7., 7.],
           [1., 0.],
           [0., 1.]])




```python
#Create stacked column-wise arrays
np.c_[a,d]
```




    array([[ 1, 10],
           [ 2, 15],
           [ 3, 20]])



#### División de matrices


```python
#Split the array horizontally at the 3rd index 
np.hsplit(a,3) 
```




    [array([1]), array([2]), array([3])]




```python
#Split the array vertically at the 2nd index
np.vsplit(c,2) 
```




    [array([[[1.5, 2. , 1. ],
             [4. , 5. , 6. ]]]), array([[[3., 2., 3.],
             [4., 5., 6.]]])]




```python

```
