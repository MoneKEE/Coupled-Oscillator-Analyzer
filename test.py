import progressbar
import time
  
  
# Function to create
def animated_marker():
      
    #widgets = ['Loading: ', progressbar.AnimatedMarker()]
    widgets = ['Loading: ', progressbar.AnimatedMarker(),' [',
         progressbar.Timer(),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
    bar = progressbar.ProgressBar(widgets=widgets,maxval=50).start()
      
    for i in range(50):
        time.sleep(0.1)
        bar.update(i)

    bar.finish()
          
# Driver's code
animated_marker()