
    \ 用 peforth 來審視本 demo 程式: 
    \ 從本範例程式所在的 directory 直接執行 python 之後 import peforth > peforth.main() 
    \ 進入 peforth 之後 include 本程式。

    --- marker ---
    
    <py>
    
    __author__ = 'Nilay Shrivastava,nilayshrivastava1729@gmail.com'

    import numpy

    class kNearestNeighbours:

        def __init__( self,k = 3 ):
            self.k = k

        def fit( self, x_train, y_train ):
            self.x = x_train
            self.y = y_train

        def predict( self, testFeatures ):
            
            # 取兩點之間的距離
            def EuclidDist( testPoint , checkPoint ):
                distance = numpy.linalg.norm( testPoint - checkPoint )
                return distance

            def closest( testPoint ):
                distArray = numpy.array( 
                    [ ( EuclidDist( testPoint , self.x[i] ), self.y[i] ) 
                      for i in range(len(self.x)) 
                    ] , 
                    dtype = [('dist', float),('lab', int)] 
                )

                # breakpoint
                # import endo; endo.l = locals(); endo.bp('111> ')
                # push({'locals':locals(),'distArray':distArray,'tag':11});ok('11> ') 
                
                distArray.sort(order = 'dist')
                majority  = {}
                for j in range(self.k):
                    if majority.get( distArray[j][1] ) == None:
                        # 沒有出現過的 key(label) distArray[j][1] 其值就是 None
                        # 第一次出現在排行榜的 label 先給 0 分
                        majority[ distArray[j][1] ] = 0 
                    else:
                        # 後續再出現就把分數增加
                        majority[ distArray[j][1] ] += 1
                # 從 majority 當中取出現次數最多的 label 

                # breakpoint
                # push({'locals':locals(),'tag':22});ok('22> ') 

                return max( majority , key = majority.get )
            # if debug<=33: ok('33> ',locals()) # breakpoint
            # 當只有一組 feature 時,傳回單一個最近的 label 作為推算結果
            # 否則一整列 feature 時,推算結果是一列 labels
            if testFeatures.ndim == 1:    #only one point to predict
                # if debug<=44: ok('44> ',locals()) # breakpoint
                return closest( testFeatures )
            else:
                # if debug<=55: ok('44> ',locals()) # breakpoint
                prediction = numpy.array( [closest(point) for point in testFeatures] )
                return prediction 


    if __name__ == '__main__' or True:

        from check import CheckClassifier

        feature_set = numpy.loadtxt(r'Fisher.csv',delimiter = ',',skiprows = 1,usecols = (1,2,3,4))
        label_set   = numpy.loadtxt(r'Fisher.csv',delimiter = ',',skiprows = 1, usecols = (0,))

        #TITANIC DATASET
        #raw_feature_set = numpy.loadtxt('titanic.txt',delimiter = ',',skiprows = 1,usecols = ())

        x_train , x_test = numpy.vsplit(feature_set,2)
        y_train , y_test = numpy.hsplit(label_set,2)

        kNN = kNearestNeighbours(5)
        kNN.fit(x_train,y_train)

        print(CheckClassifier( kNN , x_test , y_test ))
        
    </py>
    
    stop 
    
    本範例真的把 KNN classifier 打造出來了！
    class kNearestNeighbours 內含 fit() , predict() 很熟悉，正是
    classifier 改有的東西，寫得很漂亮。check.py module 也自己寫，
    提供了 CheckClassifier() 也很簡短漂亮。
    
    --
    Study EuclidDist
    
    # numpy.linalg.norm() 取向量的 norm 即其 絕對值 或 長度
    
    111> l['numpy'].linalg.norm  
    <function norm at 0x0000019F6B29D620>
    111> l['numpy'].linalg.norm((1,1))
    1.4142135623730951
    111> l['numpy'].linalg.norm((3,4))
    5.0
    111> l['numpy'].linalg.norm(((0,0),(3,4)))
    5.0
    111>
    111> print(l['numpy'].linalg.norm.__doc__)

    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : array_like
        Input array.  If `axis` is None, `x` must be 1-D or 2-D.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

        .. versionadded:: 1.10.0

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).    
    ... snip ...
    
    --
    Study distArray
    
    Study distArray 是啥東西? 
    給定一個 vector, 算出其與 fit() 而來的 feature 裡所有的 vector 的距離。
    保存在 numpy.array distArray 裡，形式為 tuple ( 距離, label) 
    numpy.array 可以指定每個 cell 的 dtype 含 名稱 與 type, 在此為 'dist',float
    與 'label',int
    
    11> constant @11
    11> @11 :> keys() .
    dict_keys(['locals', 'distArray', 'tag']) 11> @11 :> ['distArray'] .
    [( 31.67017524, 0) ( 21.02379604, 1) ( 18.70828693, 1) ( 36.95943723, 0)
     ( 15.13274595, 1) ( 10.90871211, 1) (  3.60555128, 2) ( 10.90871211, 2)
     ...snip...
     ( 29.49576241, 0) (  7.61577311, 2) ( 29.61418579, 1)] 11> @11 type .
    <class 'dict'>
    11> @11 :> ['distArray'] :> sort(order='dist') <-- sort好結果直接蓋過原來的
    None 11> @11 :> ['distArray'] .
    [(  2.44948974, 2) (  3.46410162, 2) (  3.46410162, 2) (  3.60555128, 2)
     (  3.87298335, 2) (  4.89897949, 2) (  5.56776436, 2) (  6.        , 2)
     (  6.164414  , 2) (  6.78232998, 2) (  7.        , 2) (  7.        , 2)
     (  7.34846923, 2) (  7.61577311, 2) (  9.48683298, 1) (  9.48683298, 2)
     .... snip ...
     ( 33.04542328, 0) ( 33.2565783 , 0) ( 33.86738844, 0) ( 34.13209633, 0)
     ( 34.16138171, 0) ( 36.29049462, 0) ( 36.95943723, 0)] 
    key 改用 label 重新 sort 看看。。。。
    11> @11 :> ['distArray'] :> sort(order='lab') @11 :> ['distArray'] .
    [( 28.10693865, 0) ( 29.35983651, 0) ( 29.47880595, 0) ( 29.49576241, 0)
     ( 29.71531592, 0) ( 30.0499584 , 0) ( 30.5450487 , 0) ( 31.09662361, 0)
     ( 31.25699922, 0) ( 31.38470965, 0) ( 31.67017524, 0) ( 31.96873473, 0)
    .... snip ......
     ( 13.60147051, 2) ( 14.        , 2) ( 15.19868415, 2)] 11>    
     