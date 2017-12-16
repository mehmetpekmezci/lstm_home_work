#https://youtu.be/cWomSFLAwM8

import tensorflow as tf
import csv
import glob
import sys
import os
import argparse
import sys
import tempfile
import numpy as np
import pandas as pd
import time 
import random
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13



FRAME_DATA_POINT_COUNT=20
script_dir=os.path.dirname(os.path.realpath(__file__))
main_data_dir = script_dir+'/../data/MSRAction3D/MSRAction3DSkeleton20joints'
data_dictionary=dict()


def load_data():
 global data_dictionary
 for action_no in range(20) :
   data_dictionary[action_no]=dict()
   for subject_no in range(10) :
      data_dictionary[action_no][subject_no]=dict()
      for example_no in range(3) :
         fname="a"
         if action_no < 9 :
            fname=fname+"0"+str(action_no+1)
         else :
            fname=fname+str(action_no+1)
         fname=fname+"_s"
         if subject_no < 9 :
            fname=fname+"0"+str(subject_no+1)
         else :
            fname=fname+str(subject_no+1)
         fname=fname+"_e"
         if example_no < 9 :
            fname=fname+"0"+str(example_no+1)
         else :
            fname=fname+str(example_no+1)
         fname=fname+"_skeleton.txt"
         
         if os.path.isfile(main_data_dir+"/"+fname) :
           #print("loading "+main_data_dir+"/"+fname+"  ... ")
           frames=[] ## list of frames
           data=np.array(np.genfromtxt(main_data_dir+"/"+fname ,converters = {3: lambda s: float(s or 0)}, dtype=float,delimiter="  "))
           for frame_no in range(int(data.shape[0]/FRAME_DATA_POINT_COUNT)) :
              frames.append(data[frame_no*FRAME_DATA_POINT_COUNT:(frame_no+1)*FRAME_DATA_POINT_COUNT,:] )
           data_dictionary[action_no][subject_no][example_no]=frames
         else :
           print(main_data_dir+"/"+fname+" does not exists")

def centralize_data_taking_point_7_as_center(): 
 global data_dictionary
 for action_no in range(20) :
  for subject_no in range(10) :
     for example_no in range(3) :
        if action_no in data_dictionary and subject_no in data_dictionary[action_no] and  example_no in data_dictionary[action_no][subject_no] :
         if action_no == 10 and subject_no == 5 and example_no == 2 :
           title="a"+str(action_no)+"_s" + str(subject_no) +"_e"+str(example_no)
           frames=data_dictionary[action_no][subject_no][example_no]
           centered_frames=[]
           for frameno in range(len(data_dictionary[action_no][subject_no][example_no]) -1) :
              frame=data_dictionary[action_no][subject_no][example_no][frameno]
              ##  6 = 7 -1 (0 dan saymaya basladigi icin)
              frame[:,0]=frame[:,0]-frame[6,0]
              frame[:,1]=frame[:,1]-frame[6,1]
              frame[:,2]=frame[:,2]-frame[6,2]
              frame[:,3]=0
              centered_frames.append(frame)
           data_dictionary[action_no][subject_no][example_no]=centered_frames


#              20
#               |
#           2---3---1
#         9     |     8
#       11      |      10
#      13       4      12
#               |
#               7 
#             5   6
#           14      15
#        16            17
#      18               19    



def print_data():
 global data_dictionary
 for action_no in range(20) :
  for subject_no in range(10) :
     for example_no in range(3) :
        if action_no in data_dictionary and subject_no in data_dictionary[action_no] and  example_no in data_dictionary[action_no][subject_no] :
         if action_no == 10 and subject_no == 5 and example_no == 2 :
           title="a"+str(action_no)+"_s" + str(subject_no) +"_e"+str(example_no)
           frames=data_dictionary[action_no][subject_no][example_no]
           animate(frames,title)
#         else :
#           for frame_no in range(len(data_dictionary[action_no][subject_no][example_no])) :
#            print("Action = "+str(action_no)+" --  Subject = " + str(subject_no) +"  --  Example No = "+str(example_no) + " Fame No = "+str(frame_no))
#            print(data_dictionary[action_no][subject_no][example_no][frame_no]) 



#              20
#               |
#           2---3---1
#         9     |     8
#       11      |      10
#      13       4      12
#               |
#               7 
#             5   6
#           14      15
#        16            17
#      18               19    


def animate(list_of_frames,animation_title):

  J=np.matrix([ [20  ,   1  ,   2  ,   1  ,   8 ,   10  ,   2  ,   9  ,  11  ,   3   ,  4  ,   7  ,   7  ,   5  ,   6  ,  14  ,  15  ,  16  ,  17],
      [3  ,   3  ,   3  ,   8  ,  10  ,  12  ,   9  ,  11  ,  13  ,   4   ,  7  ,   5  ,   6  ,  14  ,  15  ,  16  ,  17 ,   18  ,  19]
    ])

  # Attaching 3D axis to the figure
  fig = plt.figure()
  ax = p3.Axes3D(fig)

  # Setting the axes properties
  ax.set_xlim3d([-100.0, 400.0])
  ax.set_xlabel('X')

  ax.set_ylim3d([-100.0, 100.0])
  ax.set_ylabel('Y')

  ax.set_zlim3d([-100.0, 100.0])
  ax.set_zlabel('Z')

  title=ax.set_title(animation_title)

  #lines = [ax.plot(dat[0, : ], dat[1, : ], dat[2,  : ])[0] for dat in data]

  sequence_number=0
  S=list_of_frames[sequence_number]
  X_VECTOR=S[:,0]
  Z_VECTOR=np.subtract(np.full((len(S)), 100),S[:,1])
  Y_VECTOR=S[:,2]/4
  joints, = ax.plot(X_VECTOR,Y_VECTOR,Z_VECTOR, linestyle="", marker=".")
  lines = []

  for i in range(FRAME_DATA_POINT_COUNT-1) :
    c1=J[0,i]-1;
    c2=J[1,i]-1;
    line , =ax.plot([X_VECTOR[c1], X_VECTOR[c2]], [Y_VECTOR[c1], Y_VECTOR[c2]],[Z_VECTOR[c1], Z_VECTOR[c2]])
    lines.append(line)



    
  def update(sequence_number):
    S=list_of_frames[sequence_number]
    X_VECTOR=S[:,0]
    Z_VECTOR=np.subtract(np.full((len(S)), 100),S[:,1])
    Y_VECTOR=S[:,2]/4
    joints.set_data (X_VECTOR, Y_VECTOR)
    joints.set_3d_properties(Z_VECTOR)
    
    for i in range(FRAME_DATA_POINT_COUNT-1) :
        c1=J[0,i]-1;
        c2=J[1,i]-1;
        lines[i].set_data([X_VECTOR[c1], X_VECTOR[c2]], [Y_VECTOR[c1], Y_VECTOR[c2]]);
        lines[i].set_3d_properties([Z_VECTOR[c1], Z_VECTOR[c2]]);

    title.set_text(animation_title+' sequence={}'.format(sequence_number))
    return  tuple(lines) + (title,joints)
    #return title, joints 


  # Creating the Animation object
  #line_ani = animation.FuncAnimation(fig, update, len(list_of_frames), fargs=(frames, lines),interval=400, blit=True)
  line_ani = animation.FuncAnimation(fig, update, len(list_of_frames), interval=400, blit=True)
  line_ani.save('action_recognition.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
  plt.show()
  


class ModelNetwork:
	def __init__(self, in_size, lstm_size, num_layers, out_size, session, learning_rate=0.003, name="rnn"):
		self.scope = name

		self.in_size = in_size
		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.out_size = out_size

		self.session = session

		self.learning_rate = tf.constant( learning_rate )

		# Last state of LSTM, used when running the network in TEST mode
		self.lstm_last_state = np.zeros((self.num_layers*2*self.lstm_size,))

		with tf.variable_scope(self.scope):
			## (batch_size, timesteps, in_size)
			self.xinput = tf.placeholder(tf.float32, shape=(None, None, self.in_size), name="xinput")
			self.lstm_init_value = tf.placeholder(tf.float32, shape=(None, self.num_layers*2*self.lstm_size), name="lstm_init_value")

			# LSTM
			self.lstm_cells = [ tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
			self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=False)

			# Iteratively compute output of recurrent network
			outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.xinput, initial_state=self.lstm_init_value, dtype=tf.float32)

			# Linear activation (FC layer on top of the LSTM net)
			self.rnn_out_W = tf.Variable(tf.random_normal( (self.lstm_size, self.out_size), stddev=0.01 ))
			self.rnn_out_B = tf.Variable(tf.random_normal( (self.out_size, ), stddev=0.01 ))

			outputs_reshaped = tf.reshape( outputs, [-1, self.lstm_size] )
			network_output = ( tf.matmul( outputs_reshaped, self.rnn_out_W ) + self.rnn_out_B )

			batch_time_shape = tf.shape(outputs)
			#self.final_outputs = tf.reshape( tf.nn.softmax( network_output), (batch_time_shape[0], batch_time_shape[1], self.out_size) )
			self.final_outputs = tf.reshape( network_output, (batch_time_shape[0], batch_time_shape[1], self.out_size) )


			## Training: provide target outputs for supervised training.
			self.y_batch = tf.placeholder(tf.float32, (None, None, self.out_size))
			y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])

			#self.cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_batch_long) )
			self.cost = tf.losses.mean_squared_error(y_batch_long,network_output) 

			self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cost)

			#self.train_op  = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)




	## Input: X is a single element, not a list!
	def run_step(self, x, init_zero_state=True):
		## Reset the initial state of the network.
		if init_zero_state:
			init_value = np.zeros((self.num_layers*2*self.lstm_size,))
		else:
			init_value = self.lstm_last_state

		out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state], feed_dict={self.xinput:[x], self.lstm_init_value:[init_value]   } )

		self.lstm_last_state = next_lstm_state[0]

		return out[0][0]


	## xbatch must be (batch_size, timesteps, input_size)
	## ybatch must be (batch_size, timesteps, output_size)
	def train_batch(self, xbatch, ybatch):
		init_value = np.zeros((xbatch.shape[0], self.num_layers*2*self.lstm_size))

		cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.xinput:xbatch, self.y_batch:ybatch, self.lstm_init_value:init_value   } )

		return cost


load_data()
#centralize_data_taking_point_7_as_center()
#print_data()






ckpt_file = "saved.model.ckpt"
test_action_no=10
test_subject_no=5
test_example_no=2
TEST_FRAME = data_dictionary[test_action_no][test_subject_no][test_example_no][0]
TOTAL_ACTIONS=20
TOTAL_SUBJECTS=10
TOTAL_EXAMPLE=3
NUMBER_OF_TRAINING_STEPS= 100 * TOTAL_ACTIONS * TOTAL_SUBJECTS * TOTAL_EXAMPLE
in_size = out_size = FRAME_DATA_POINT_COUNT * 3 # 20 data points times 3 (X,Y,Z)
lstm_size = 256
num_layers = 2
NUMBER_OF_TEST_FRAMES = 25 # Number of test human position frames to generate after training the network
time_steps = 100


## Initialize the network
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

net = ModelNetwork(in_size = in_size,lstm_size = lstm_size,num_layers = num_layers,
		   out_size = out_size,session = sess,learning_rate = 0.0001,
		   name = "human_position_rnn_network")

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())




#print "Usage:"
#print '\t\t ', sys.argv[0], ' [ckpt model to load] [prefix, e.g., "The "]'
#if len(sys.argv)>=2:
#	ckpt_file=sys.argv[1]
#if len(sys.argv)==3:
#	TEST_FRAME = sys.argv[2]


start_time=time.time()

## 1) TRAIN THE NETWORK
if not os.path.isfile(ckpt_file) :
 for trainingStep in range(NUMBER_OF_TRAINING_STEPS):
  action_no=random.randint(0, 19)
  subject_no=random.randint(0, 9)
  example_no=random.randint(0, 2)
  if action_no != test_action_no and subject_no != test_subject_no and  example_no != test_example_no :
    if action_no in data_dictionary and subject_no in data_dictionary[action_no] and  example_no in data_dictionary[action_no][subject_no] :
      #print("Learning Frames of Action No ="+str(action_no)+"  Subject No = "+str(subject_no)+" Example No = "+str(example_no))
      
      #batch_size=len(data_dictionary[action_no][subject_no][example_no])
      batch_size=1
      batch = np.zeros((batch_size, time_steps, in_size))
      batch_y = np.zeros((batch_size, time_steps, in_size))
      for i in range(len(data_dictionary[action_no][subject_no][example_no])) :
        input_frame=data_dictionary[action_no][subject_no][example_no][i]
        input_frame=np.reshape(input_frame[:,0:3], (60))
        if i == len(data_dictionary[action_no][subject_no][example_no])-1 :
          output_frame=data_dictionary[action_no][subject_no][example_no][0]
        else :
          output_frame=data_dictionary[action_no][subject_no][example_no][i+1]

        output_frame=np.reshape(output_frame[:,0:3], (60))
        batch[:, i, :] = input_frame
        batch_y[:, i, :] = output_frame

      cst = net.train_batch(batch, batch_y)
      if trainingStep%100 == 0  :   
        print ("Training Time = ",str(time.time()-start_time),"  Training Cost = ", cst, "  Training Step =",trainingStep)
        start_time=time.time()
     


 saver.save(sess, "saved.model.ckpt")





## 2) GENERATE NUMBER_OF_TEST_FRAMES FRAMES USING THE TRAINED NETWORK

if os.path.isfile(ckpt_file) :
	saver.restore(sess, ckpt_file)

generated_frames=[]
generated_frames.append(TEST_FRAME)
TEST_FRAME_INPUT=np.reshape(TEST_FRAME[:,0:3], (1,60))

for i in range(NUMBER_OF_TEST_FRAMES):
    print(i)
    input_=TEST_FRAME_INPUT
    output_ = net.run_step( input_ , i==0)
   # output_=np.reshape(output_, (60))
    print(output_.shape)
    TEST_FRAME_INPUT=np.reshape(output_, (1,60))
    generated_frames.append(np.reshape(output_, (20,3)))
    


print(len(generated_frames))
print(generated_frames)
animate(generated_frames,"generated_frames")







