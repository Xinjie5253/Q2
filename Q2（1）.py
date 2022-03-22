import numpy as np # this one is to do manipulate arrays 
import scipy as sp
import scipy.integrate as spi
import matplotlib.pyplot as plt  # this one is to produce nice graphs and plots


class PidController:

  def __init__(self, kp, ki, kd, ts):
      # TODO: initialise all attributes of this class
      # The arguments kp, ki, kd are the continuous-time 
      # gains of the PID controller and ts is the sampling
      # time
      self.__kp = kp
      self.__ki = ki * ts
      self.__kd = kd / ts
      self.__sum_errors = 0
      self.__previous_error = None
     

  def control(self, y, set_point=0):
      # P for Proportional
      error = set_point - y
      u = self.__kp * error

      # I for Integral
      self.__sum_errors += error
      u += self.__ki * self.__sum_errors

      # D for Derivatives
      if self.__previous_error is not None:
          diff_errors = error - self.__previous_error
          u += self.__kd * diff_errors
      
      self.__previous_error = error
      return u



brain = PidController(kp=1, kd=0.1, ki=0.5, ts=0.01)




class Car:

    def __init__(self, 
             length=2.3,
             velocity=1,
             x_pos_init=0, y_pos_init=0.3, pose_init=0):
        self.__length = length
        self.__velocity = velocity
        self.__x = x_pos_init
        self.__y = y_pos_init
        self.__pose = pose_init

    def y(self):
        return self.__y

    def move(self, steering_angle, dt):
        # This method computes the position and orientation (pose) 
        # of the car after time `dt` starting from its current 
        # position and orientation by solving an IVP
        #
        def bicycle_model(_t, z):
            # Define system dynamics [given in (2.1a-c)]
            # [v*cos(θ), v*sin(θ), (v/L)*tan(u)]
            # z = (x, y, θ)
            theta=z[2]
            return [self.__velocity*np.cos(theta),
                    self.__velocity*np.sin(theta),
                    self.__velocity*np.tan(steering_angle+np.deg2rad(1))/self.__length] # add disturbance w

        sol = spi.solve_ivp(bicycle_model, 
                            [0, dt],
                            [self.__x, self.__y, self.__pose], 
                            t_eval=np.linspace(0, dt, 2))
        new_state = sol.y[:, -1]
        self.__x = new_state[0]
        self.__y = new_state[1]
        self.__pose = new_state[2]

    def velocity(self):
        return self.__velocity
   
    def state(self):
        return [self.__x, self.__y, self.__pose]




a = 0
b = 50
n_points = 2000

# build arrays in 2 dimensions to store data simulated at different kp
x_p =np.zeros((4,n_points+1,))
y_p =np.zeros((4,n_points+1,))
h = (b-a)/n_points
kp=0.1
for a in range(4):
    kp=kp+0.1 # increase kp at 0.1 for each loop
    brain = PidController(kp=kp, kd=0, ki=0, ts=0.025)
    henry = Car(length=2.3, velocity=5)
    for i in range(n_points+1):

        u=brain.control(y=henry.y())
        henry.move(steering_angle=np.deg2rad(1)+u, dt=0.025) # dt= 50/n_points
        z=henry.state()
        x_p[a,i]=z[0]
        y_p[a,i]=z[1]
for b in range(4):
  plt.plot(x_p[b],y_p[b])
plt.legend(('kp=0.2','kp=0.3','kp=0.4','kp=0.5'))
plt.rcParams['font.size'] = '30'
plt.xlabel('x')
plt.ylabel('y')
plt.show()
        


    


    

    

