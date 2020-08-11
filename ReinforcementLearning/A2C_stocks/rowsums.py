def hourglassSum(arr): 
    rows = len(arr) 
    cols = len(arr[0]) 
  
    sumvals = [] 
    for c in range(cols-2): 
        for r in range(rows-2): 
            sumval  = sum([arr[r][ci] for ci in range(c,c+3)]) 
            sumval += arr[r+1][c+1] 
            sumval += sum([arr[r+2][ci] for ci in range(c,c+3)]) 
            sumvals.append(sumval) 

    return max(sumvals)

if __name__ == '__main__':

    arr = [[1, 1, 1, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [1, 1, 1, 0, 0, 0],
           [0, 0, 2, 4, 4, 0],
           [0, 0, 0, 2, 0, 0],
           [0, 0, 1, 2, 4, 0]]
    
    print(hourglassSum(arr))

    arr = [[-9, -9, -9, 1, 1, 1],
           [0, -9, 0, 4, 3, 2],
           [-9, -9, -9, 1, 2, 3],
           [0, 0, 8, 6, 6, 0],
           [0, 0, 0, -2, 0, 0],
           [0, 0, 1, 2, 4, 0]]
    
    print(hourglassSum(arr))

