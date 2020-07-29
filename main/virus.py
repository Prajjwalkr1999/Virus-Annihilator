import sys, pygame, random,time

# directkeys.py is taken from https://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# inspired from pyimagesearch ball tracking https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from directkeys import  W, A, S, D
from directkeys import PressKey, ReleaseKey 


# define the lower and upper boundaries of the "orange" object in the HSV color space

orangeLower = np.array([35, 90, 102])
orangeUpper = np.array([102, 255, 255])

vs = VideoStream(src=0).start()
# allow the camera or video file to warm up
time.sleep(2.0)
initial = True
flag = False
current_key_pressed = set()
circle_radius = 30
windowSize = 160
lr_counter = 0

class Wall():

    def __init__(self):
        self.brick = pygame.image.load("brick.png").convert()
        brickrect = self.brick.get_rect()
        self.bricklength = brickrect.right - brickrect.left       
        self.brickheight = brickrect.bottom - brickrect.top             

    def build_wall(self, width):        
        xpos = 0
        ypos = 60
        adj = 0
        self.brickrect = []
        for i in range (0, 62):           
            if xpos > width:
                if adj == 0:
                    adj = self.bricklength / 2
                else:
                    adj = 0
                xpos = -adj
                ypos += self.brickheight
                
            self.brickrect.append(self.brick.get_rect())    
            self.brickrect[i] = self.brickrect[i].move(xpos, ypos)
            xpos = xpos + self.bricklength

class obstlacles:
    def __init__(self):
        self.arr = []

    def add(self,n):
        self.arr.clear()
        while n:
            x = [random.choice(range(10, 630)) , random.choice(range(200,400)), random.choice((+1,-1))]
            self.arr.append(x)
            n -= 1;


def obs_collide():
    global score,xspeed_init,yspeed_init,width
    sz = len(obs)
    # mn = obs[0][0]*screen_width + obb.arr[0][1]
    # mx = obb.arr[sz-1][0]*screen_width + obb.arr[sz-1][1]
    l = 0;
    e = sz-1;
    print(obs)
    while(l<=e):
        mid = int((l+e)/2)

        cur = ballrect.x*width + ballrect.y
        x = obs[mid].x*width + obs[mid].y
        if(x > cur):
            if ballrect.colliderect(obs[mid]):
                score+=20
            break
            l = mid+1
        elif(x < cur):
            if ballrect.colliderect(obs[mid]):
                score+=20
            break
            e = mid-1
        else:
            score+=20
            break

def object_collide():
    global score
    for i in range(len(obs)-1):
        if ballrect.colliderect(obs[i]):
            if obb.arr[i][2]>=1:
                score+=20
            else:
                score-=20
            obs.pop(i)
            obb.arr.pop(i)

xspeed_init = 6
yspeed_init = 6
max_lives = 100
bat_speed = 3
score = 0 
bgcolour = 0,0,0  # darkslategrey        
size = width, height = 640, 480
obs=[]

pygame.init()            
screen = pygame.display.set_mode(size)
#screen = pygame.display.set_mode(size, pygame.FULLSCREEN)

bat = pygame.image.load("bat.png").convert()
batrect = bat.get_rect()

ball = pygame.image.load("ball.png").convert()
ball.set_colorkey((255, 255, 255))
ballrect = ball.get_rect()

pong = pygame.mixer.Sound('Blip_1-Surround-147.wav')
pong.set_volume(10)        

wall = Wall()
wall.build_wall(width)

# Initialise ready for game loop
batrect = batrect.move((width / 2) - (batrect.right / 2), height - 20)
ballrect = ballrect.move(width / 2, height / 2)       
xspeed = xspeed_init
yspeed = yspeed_init
lives = max_lives
clock = pygame.time.Clock()
pygame.key.set_repeat(1,30)       
pygame.mouse.set_visible(0)       # turn off mouse pointer

obb = obstlacles()
obb.add(10)

prev_time = time.perf_counter()

while 1:

    keyPressed = False

    keyPressed_lr = False
    # grab the current frame
    frame = vs.read()
    frame = cv2.flip(frame,1)
    height,width = frame.shape[:2]
 
    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # crteate a mask for the orange color and perform dilation and erosion to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
 
    # find contours in the mask and initialize the current
    # (x, y) center of the orange object

    # divide the frame into two halves so that we can have one half control the acceleration/brake 
    # and other half control the left/right steering.
    left_mask = mask[:,0:width,]
    # right_mask = mask[:,width:,]

    #find the contours in the left and right frame to find the center of the object
    cnts_left = cv2.findContours(left_mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts_left = imutils.grab_contours(cnts_left)
    center_left = None

 
    # only proceed if at least one contour was found
    if len(cnts_left) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and centroid
        c = max(cnts_left, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        # find the center from the moments 0.000001 is added to the denominator so that divide by 
        # zero exception doesn't occur
        center_left = (int(M["m10"] / (M["m00"]+0.000001)), int(M["m01"] / (M["m00"]+0.000001)))
    
        # only proceed if the radius meets a minimum size
        if radius > circle_radius:
            # draw the circle and centroid on the frame,
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center_left, 5, (0, 0, 255), -1)

            #the window size is kept 160 pixels in the center of the frame(80 pixels above the center and 80 below)
            if center_left[1] < (height/2 - windowSize//2):
                cv2.putText(frame,'UP',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                PressKey(W)
                current_key_pressed.add(A)
                keyPressed = True
                keyPressed_lr = True
                # player_speed -= 2

            elif center_left[1] > (height/2 + windowSize//2):
                cv2.putText(frame,'DOWN',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                PressKey(S)
                current_key_pressed.add(D)
                keyPressed = True
                keyPressed_lr = True
                # player_speed += 2



 
    # show the frame to our screen
    frame_copy = frame.copy()
    frame_copy = cv2.rectangle(frame_copy,(0,height//2 - windowSize//2),(width,height//2 + windowSize//2),(255,0,0),2)
    cv2.imshow("Frame", frame_copy)

       #We need to release the pressed key if none of the key is pressed else the program will keep on sending
    # the presskey command 
    if not keyPressed and len(current_key_pressed) != 0:
        for key in current_key_pressed:
            ReleaseKey(key)
        current_key_pressed = set()

    #to release keys for left/right with keys of up/down remain pressed   
    if not keyPressed_lr and ((A in current_key_pressed) or (D in current_key_pressed)):
        if A in current_key_pressed:
            ReleaseKey(A)
            current_key_pressed.remove(A)
            # player_speed += 2

        elif D in current_key_pressed:
            ReleaseKey(D)
            current_key_pressed.remove(D)
            # player_speed -= 2

    key = cv2.waitKey(1) & 0xFF

    # 60 frames per second
    clock.tick(60)

    # process key presses
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                sys.exit()
            if event.key == pygame.K_w:                        
                batrect = batrect.move(-bat_speed, 0)     
                if (batrect.left < 0):                           
                    batrect.left = 0      
            if event.key == pygame.K_s:                    
                batrect = batrect.move(bat_speed, 0)
                if (batrect.right > width):                            
                    batrect.right = width

    new_time = time.perf_counter()
    if new_time - prev_time > 5.0:
        prev_time = new_time
        obb.arr.clear()
        obb.add(5)
        obs.clear()
        for ele in obb.arr:
            x = pygame.Rect(ele[0], ele[1], 30,30)
            obs.append(x)

    object_collide()


    # check if bat has hit ball    
    if ballrect.bottom >= batrect.top and \
       ballrect.bottom <= batrect.bottom and \
       ballrect.right >= batrect.left and \
       ballrect.left <= batrect.right:
        yspeed = -yspeed                
        pong.play(0)                
        offset = ballrect.center[0] - batrect.center[0]                          
        # offset > 0 means ball has hit RHS of bat                   
        # vary angle of ball depending on where ball hits bat                      
        if offset > 0:
            if offset > 30:  
                xspeed = 7
            elif offset > 23:                 
                xspeed = 6
            elif offset > 17:
                xspeed = 5 
        else:  
            if offset < -30:                             
                xspeed = -7
            elif offset < -23:
                xspeed = -6
            elif xspeed < -17:
                xspeed = -5     
              
    # move bat/ball
    ballrect = ballrect.move(xspeed, yspeed)
    if ballrect.left < 0 or ballrect.right > width:
        xspeed = -xspeed                
        pong.play(0)            
    if ballrect.top < 0:
        yspeed = -yspeed                
        pong.play(0)               

    # check if ball has gone past bat - lose a life
    if ballrect.top > height:
        lives -= 1
        # start a new ball
        xspeed = xspeed_init
        rand = random.random()                
        if random.random() > 0.5:
            xspeed = -xspeed 
        yspeed = yspeed_init            
        ballrect.center = width * random.random(), height / 3                                
        if lives == 0:                    
            msg = pygame.font.Font(None,70).render("Game Over", True, (0,255,255), bgcolour)
            msgrect = msg.get_rect()
            msgrect = msgrect.move(width / 2 - (msgrect.center[0]), height / 3)
            screen.blit(msg, msgrect)
            pygame.display.flip()
            # process key presses
            #     - ESC to quit
            #     - any other key to restart game
            while 1:
                # keyPressed = False

                # keyPressed_lr = False
                # # grab the current frame
                # frame = vs.read()
                # frame = cv2.flip(frame,1)
                # height,width = frame.shape[:2]
             
                # # resize the frame, blur it, and convert it to the HSV color space
                # frame = imutils.resize(frame, width=600)
                # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
                # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                
                # # crteate a mask for the orange color and perform dilation and erosion to remove any small
                # # blobs left in the mask
                # mask = cv2.inRange(hsv, orangeLower, orangeUpper)
                # mask = cv2.erode(mask, None, iterations=2)
                # mask = cv2.dilate(mask, None, iterations=2)
             
                # # find contours in the mask and initialize the current
                # # (x, y) center of the orange object

                # # divide the frame into two halves so that we can have one half control the acceleration/brake 
                # # and other half control the left/right steering.
                # left_mask = mask[:,0:width,]
                # # right_mask = mask[:,width:,]

                # #find the contours in the left and right frame to find the center of the object
                # cnts_left = cv2.findContours(left_mask.copy(), cv2.RETR_EXTERNAL,
                #     cv2.CHAIN_APPROX_SIMPLE)
                # cnts_left = imutils.grab_contours(cnts_left)
                # center_left = None

             
                # # only proceed if at least one contour was found
                # if len(cnts_left) > 0:
                #     # find the largest contour in the mask, then use
                #     # it to compute the minimum enclosing circle and centroid
                #     c = max(cnts_left, key=cv2.contourArea)
                #     ((x, y), radius) = cv2.minEnclosingCircle(c)
                #     M = cv2.moments(c)
                #     # find the center from the moments 0.000001 is added to the denominator so that divide by 
                #     # zero exception doesn't occur
                #     center_left = (int(M["m10"] / (M["m00"]+0.000001)), int(M["m01"] / (M["m00"]+0.000001)))
                
                #     # only proceed if the radius meets a minimum size
                #     if radius > circle_radius:
                #         # draw the circle and centroid on the frame,
                #         cv2.circle(frame, (int(x), int(y)), int(radius),
                #             (0, 255, 255), 2)
                #         cv2.circle(frame, center_left, 5, (0, 0, 255), -1)

                #         #the window size is kept 160 pixels in the center of the frame(80 pixels above the center and 80 below)
                #         if center_left[1] < (height/2 - windowSize//2):
                #             cv2.putText(frame,'UP',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                #             PressKey(W)
                #             current_key_pressed.add(A)
                #             keyPressed = True
                #             keyPressed_lr = True
                #             # player_speed -= 2

                #         elif center_left[1] > (height/2 + windowSize//2):
                #             cv2.putText(frame,'DOWN',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                #             PressKey(S)
                #             current_key_pressed.add(D)
                #             keyPressed = True
                #             keyPressed_lr = True
                #             # player_speed += 2


             
                # # show the frame to our screen
                # frame_copy = frame.copy()
                # frame_copy = cv2.rectangle(frame_copy,(0,height//2 - windowSize//2),(width,height//2 + windowSize//2),(255,0,0),2)
                # cv2.imshow("Frame", frame_copy)

                #    #We need to release the pressed key if none of the key is pressed else the program will keep on sending
                # # the presskey command 
                # if not keyPressed and len(current_key_pressed) != 0:
                #     for key in current_key_pressed:
                #         ReleaseKey(key)
                #     current_key_pressed = set()

                # #to release keys for left/right with keys of up/down remain pressed   
                # if not keyPressed_lr and ((A in current_key_pressed) or (D in current_key_pressed)):
                #     if A in current_key_pressed:
                #         ReleaseKey(A)
                #         current_key_pressed.remove(A)
                #         # player_speed += 2

                #     elif D in current_key_pressed:
                #         ReleaseKey(D)
                #         current_key_pressed.remove(D)
                #         # player_speed -= 2

                # key = cv2.waitKey(1) & 0xFF

                restart = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            sys.exit()
                        if not (event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT):                                    
                            restart = True      
                if restart:                   
                    screen.fill(bgcolour)
                    wall.build_wall(width)
                    lives = max_lives
                    score = 0
                    break
    
    if xspeed < 0 and ballrect.left < 0:
        xspeed = -xspeed                                
        pong.play(0)

    if xspeed > 0 and ballrect.right > width:
        xspeed = -xspeed                               
        pong.play(0)
   
    # check if ball has hit wall
    # if yes yhen delete brick and change ball direction
    index = ballrect.collidelist(wall.brickrect)       
    if index != -1: 
        if ballrect.center[0] > wall.brickrect[index].right or \
           ballrect.center[0] < wall.brickrect[index].left:
            xspeed = -xspeed
        else:
            yspeed = -yspeed                
        pong.play(0)              
        wall.brickrect[index:index + 1] = []
        score += 10
                  
    screen.fill(bgcolour)
    scoretext = pygame.font.Font(None,40).render(str(score), True, (0,255,255), bgcolour)
    scoretextrect = scoretext.get_rect()
    scoretextrect = scoretextrect.move(width - scoretextrect.right, 0)
    screen.blit(scoretext, scoretextrect)

    # Visuals 
    # screen.fill((0,0,0))
    for i in range(len(obs)-1):
        if obb.arr[i][2] == 1:
            pygame.draw.rect(screen, (228, 227, 227), obs[i])
        else:
            pygame.draw.rect(screen, (132, 169, 172), obs[i])
    for i in range(0, len(wall.brickrect)):
        screen.blit(wall.brick, wall.brickrect[i])    

    # if wall completely gone then rebuild it
    if wall.brickrect == []:              
        wall.build_wall(width)                
        xspeed = xspeed_init
        yspeed = yspeed_init                
        ballrect.center = width / 2, height / 3
 
    screen.blit(ball, ballrect)
    screen.blit(bat, batrect)

        # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

    pygame.display.flip()

vs.stop() 
# close all windows
cv2.destroyAllWindows()
# if __name__ == '__main__':
#     br = Breakout()
#     br.main()