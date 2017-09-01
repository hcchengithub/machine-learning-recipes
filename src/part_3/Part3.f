
    --- marker ---
    
    <py>
    # Part 3 - What Makes a Good Feature? - https://youtu.be/N9fDIAflCMY

    # Good features are informative, independent, and simple. We'll introduce these 
    # concepts by using a histogram to visualize a feature from a toy dataset.

    import matplotlib.pyplot as plt
    import numpy as np

    # Create population of 1000 dog, 50/50 greyhound/labrador
    greyhounds = 500
    labs = 500

    # Assume greyhounds are normally 28" tall
    # Assume labradors are normally 24" tall
    # Assume normal distribution of +/- 4"
    grey_height = 28 + 4 * np.random.randn(greyhounds) # 查看 randn() 用法 locals :> ['np'] :> random.randn.__doc__ .
    lab_height = 24 + 4 * np.random.randn(labs) # 查看 
    
    # 111>locals :> ['lab_height'].shape .  ==> (500,) 
    # 111>locals :> ['grey_height'].shape . ==> (500,) 

    # Greyounds - red, labradors - blue
    plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
    plt.show()

    # Independent features are best
    # Avoid redundant features (height in inch AND height in cm)

    # Ideal features are:
    # Informative
    # Independent
    # Simple
    
    push(locals());ok('111>')
    </py>

    \ 單獨畫一張黃色的 grey_height
    \ 111>locals :: ['plt'].hist(v('locals')['grey_height'],color='y')
    \ 111>locals :: ['plt'].show()
    
    stop
    
    本課老師的意思是光從高度來分辨兩種犬類，有部分重疊而無法分辨。
    如上 Ideal features 的指標所陳。




    


    