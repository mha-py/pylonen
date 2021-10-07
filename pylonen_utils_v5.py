import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
        

cos = np.cos
sin = np.sin
pi = np.pi
deg = pi/180

# Toggle thick borders or line borders
BORDER_THICKNESS = 0.08
#BORDER_THICKNESS = 0.0



def rotate(v, alpha):
    'Rotates vector v to the left by angle alpha'
    rot = np.array([[cos(alpha), -sin(alpha)],
                    [sin(alpha),  cos(alpha)]])
    return np.matmul(rot, v)

def ortho(v):
    norm = np.sqrt((v**2).sum())
    #assert norm > 0
    if norm == 0: return v
    return np.array([-v[1], v[0]]) / norm

def normed(v):
    norm = np.sqrt((v**2).sum())
    #assert norm > 0
    if norm == 0: return v
    return v / np.sqrt((v**2).sum())

def norm(v):
    return np.sqrt((v**2).sum())

def min_none(a1, a2):
    # Helper function needed in 'route'
    if a1 is None:
        return a2
    if a2 is None:
        return a1
    return min(a1, a2)




'''Line und Ray Basisklassen
Dies sind die Basisklassenfür eine Linien und einen Strahl. Für beide gibt es einen Befehl zum Plotten. Die Linienklasse verfügt über eine Schnittpunkterkennung mit Strahlen.'''

from numba import jit, njit, float64, void

'''
@njit((float64[:])(float64[:], float64[:], float64[:], float64[:], float64[:]), nogil=True)
def intersection_line_ray_easy(line_a, line_b, line_bma, ray_p, ray_d):
    'Plane function giving the same result as Line.intersection(), optimizable by numba.jit'
    M = np.stack((line_bma, -ray_d), 1)
    if np.linalg.det(M) == 0:
        return np.array([-1., -1.])
    return np.linalg.solve(M, ray_p-line_a)''';


class Line:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.bma = b-a
        
    def intersection(self, ray, plot=False):
        det = -self.bma[0]*ray.d[1]+self.bma[1]*ray.d[0]
        if det == 0:
            return None, None
        t = -ray.d[1]*(ray.p[0]-self.a[0]) + ray.d[0]*(ray.p[1]-self.a[1])
        s = -self.bma[1]*(ray.p[0]-self.a[0]) + self.bma[0]*(ray.p[1]-self.a[1])
        t /= det
        s /= det
        if plot:
            plt.scatter((ray.p[0]+s*ray.d[0],), (ray.p[1]+s*ray.d[1],))
        
        #return t, s
        if 0 <= t <= 1 and 0 <= s:
            return t, s
        else:
            return None, None
    
        # LGS p+s*d = a+t*(b-a)
        # --> p-a = t*(b-a)-s*d
        # --> p-a = ((b-a|-d))*(t,s)^t
        # LGS per Hand lösen:
        # M00 = bx-ax
        # M10 = by-ay
        # M01 = -dx
        # M11 = -dy
        # det = M00*M11-M01*M10 = -(bx-ax)*dy+(by-ay)*dx
        # Minv00 = +M11 = -dy
        # Minv10 = -M10 = -(by-ay)
        # Minv01 = -M01 = +dx
        # Minv11 = +M00 = bx-ax
        # t = Minv00*(p-a)_x + Minv01*(p-a)_y
    
    def plot(self):
        plt.plot([self.a[0], self.b[0]], [self.a[1], self.b[1]], 'x--')
        
class Ray:
    def __init__(self, p, d):
        self.p = p
        self.d = d
        
    def plot(self):
        plt.plot([self.p[0], self.p[0]+self.d[0]], [self.p[1], self.p[1]+self.d[1]], 'o--')
        
        
        
        
'''Linienzugklasse
Ein Linienzug ist eine Verkettung von Linien. Diese Klasse verfügt ebenfalls über eine Funktion zum Plotten und eine Erkennung für Schnittpunkte mit Strahlen.'''

# Folgende Funktion wird gewrappt, um mehrere Rays mit einer Polyline kollidieren zu lassen
@njit(void(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:]))
def intersections_lines_rays(lines_a, lines_bma, rays_p, rays_d, ss):
    for i in range(len(rays_p)):
        for j in range(len(lines_a)):
            M = np.stack((lines_bma[j], -rays_d[i]), 1)
            if np.linalg.det(M) == 0:
                ss[i,j] = 9999999
                continue
            t, s = np.linalg.solve(M, rays_p[i]-lines_a[j])
            if not 0<=t<=1:
                ss[i,j] = 9999999
            else:
                ss[i,j] = s
        
class Polyline:
    
    def __init__(self):
        self.pts = []
        self.lines = []
        
    def add(self, p):
        if len(self.pts) >= 1:
            self.lines.append(Line(self.pts[-1], p))
            self.lines_a = np.stack([l.a for l in self.lines])
            self.lines_bma= np.stack([l.bma for l in self.lines])
        self.pts.append(p)
        self.ss = 999999.*np.ones((99, len(self.lines))) # Buffer für die intersection funktion
        
    def intersection(self, ray, plot=False):
        s_arr = []
        t_arr = []
        i_arr = []
        for i, l in enumerate(self.lines):
            t, s = l.intersection(ray)
            if s and s >= 0 and t and 0 <= t <= 1:
                t_arr.append(t)
                s_arr.append(s)
                i_arr.append(i)
        if len(s_arr) == 0:
            return None, None
        s = np.min(s_arr)
        k = np.argmin(s_arr)
        t = t_arr[k]
        i = i_arr[k]
        if plot:
            plt.scatter((ray.p[0]+s*ray.d[0],), (ray.p[1]+s*ray.d[1],))
        return i + t, s
        
    def intersection_optimized(self, rays_p, rays_d):
        intersections_lines_rays(self.lines_a, self.lines_bma, rays_p, rays_d, self.ss)
        return self.ss.min(1)
    
            
    def plot(self, c=None):
        plt.plot([p[0] for p in self.pts], [p[1] for p in self.pts], '-', c=c, lw=2)
        
    def close(self):
        pass
        
        
        
class Polyline_thick:
    
    def __init__(self, th=0.1):
        'Creates a polyline with thickness th.'
        self.th = th
        self.pts = []
        self.polyline = None
        
    def add(self, p):
        self.pts.append(p)
        
    def close(self):
        'Erstellt wird die Berandung der dünnen Linie, d. h. die Polyline_thick ist im Grund eine berandete, ausgedehnte Linie'
        self.polyline = Polyline()
        
        left  = []
        right = []
        
        
        # Erster Punkt adhoc
        p0 = 1*self.pts[0]
        p1 = 1*self.pts[1]
        o = ortho(p1-p0)
        left += [p0 + self.th/2*o]
        right += [p0 - self.th/2*o]
        dold = normed(p1-p0)
        
        for i in range(1, len(self.pts)-1):
            p0 = 1*self.pts[i]
            p1 = 1*self.pts[i+1]
            d = normed(p1-p0)
            o = ortho(p1-p0)
            if not np.allclose(p0, p1):
                alpha = -np.arcsin(np.dot(dold, o))
                tan = np.tan(alpha/2)
                pt_left  = p0 + self.th/2 * (o + tan*d)
                pt_right = p0 - self.th/2 * (o + tan*d)
                left  += [pt_left]
                right += [pt_right]
            dold = d
            
        # Letzter Punkt adhoc
        p0 = self.pts[-1]
        left += [p0 + self.th/2*o]
        right += [p0 - self.th/2*o]
        
        
        # Create polyline with all these points
        for pt in left:
            self.polyline.add(pt)
        for pt in reversed(right):
            self.polyline.add(pt)
        self.polyline.add(left[0])
        
        
    def intersection(self, ray, plot=False):
        return self.polyline.intersection(ray, plot)
    
    def intersection_optimized(self, rays_p, rays_d):
        return self.polyline.intersection_optimized(rays_p, rays_d)
                          
    def plot(self, c='gray'):
        p = patches.Polygon(self.polyline.pts, facecolor=c, zorder=-100)
        plt.gca().add_patch(p)
        #self.polyline.plot() # for debugging
        
        
        


'''Routenklasse
Durch diese Klasse kann eine Strecke erzeugt werden. Die linke und rechte Fahrbahnbegrenzung sind Poylines.'''
    
class Route:
    
    def __init__(self, start_p=None, start_angle=0, start_width=1, border_th=0.1):
        self.delta = 0.5 # distance of steps while creating the route
        self.p = start_p if start_p is not None else np.array([0.,0.])
        self.d = rotate(np.array([1.,0.]), start_angle)
        self.w = start_width
        self.c = 'black'
        self.p0 = self.p # (für self.close)
        self.d0 = self.d # (für self.close)
        self.middle   = Polyline()
        if border_th > 0.:
            self.border_l = Polyline_thick(border_th)
            self.border_r = Polyline_thick(border_th)
        else:
            self.border_l = Polyline()
            self.border_r = Polyline()
        self.progress_lines = [] # better name?
        self.add() # add this state to the borders
        
    def add(self):
        'Takes actual p, d, w and records the border lines'
        o = ortho(self.d)
        self.middle.add(self.p)
        self.border_l.add(self.p+self.w/2*o)
        self.border_r.add(self.p-self.w/2*o)
        self.progress_lines.append(Line(self.p+self.w/2*o, self.p-self.w/2*o))
        
    def plot(self, all=False, lw_factor=1.):
        f = lw_factor
        #for border in [self.border_l, self.border_r]:
        #    plt.plot([p[0] for p in border.pts], [p[1] for p in border.pts], '-', c=self.c, lw=f*1.5)
        self.border_l.plot(self.c)
        self.border_r.plot(self.c)
        if all:
            plt.plot([p[0] for p in self.middle.pts], [p[1] for p in self.middle.pts], '-', c='blue', lw=f*1.5)
            for l in self.progress_lines:
                plt.plot([l.a[0], l.b[0]], [l.a[1], l.b[1]], '--', c='orange', lw=f*1)
                
        plt.axis('equal')
    
    def straight_old(self, distance):
        # Gab probleme beim close, wenn distance kein vielfaches von delta ist, dann wurde die Kurve nicht richtig geschlossen
        'Adds a straight part to the route'
        n = int(distance/self.delta)
        for i in range(n):
            self.p = self.p + self.delta * self.d
            self.add()
    
    def straight(self, distance):
        'Adds a straight part to the route'
        n = int(distance//self.delta)
        n = max(1, n)
        dlt = distance/n
        for i in range(n):
            self.p = self.p + dlt * self.d
            self.add()
            
    def curve_circle(self, angle, distance):
        'Adds a curve to the route'
        n = int(distance//self.delta)
        n = max(1, n)
        dlt = distance/n
        for i in range(n):
            d_old = self.d
            self.d = rotate(self.d, angle/n)
            self.p = self.p + dlt * normed((self.d+d_old)/2)
            self.add()
            
    def curve(self, angle, distance, f=None):
        'Adds a curve to the route'
        n = int(distance//self.delta)
        n = max(1, n)
        dlt = distance/n
        if f is None:
            f = lambda x: 1. # constant function
        tinyangles = np.array([ f((i+0.5)/n) for i in range(n) ], 'float')
        tinyangles *= angle / tinyangles.sum()
        for i in range(n):
            d_old = self.d
            self.d = rotate(self.d, tinyangles[i])
            self.p = self.p + dlt * normed((self.d+d_old)/2)
            self.add()
            
    def curve_edgy(self, angle):
        'Adds a curve with an edge (ad hoc)'
        o = ortho(self.d)
        tan = np.tan(abs(angle)/2)
        if angle >= 0:
            p_l = self.p + self.w/2*o
            p_r = self.p - self.w/2*o + self.w*tan*self.d
        else:
            p_l = self.p + self.w/2*o + self.w*tan*self.d
            p_r = self.p - self.w/2*o
        self.border_l.add(p_l)
        self.border_r.add(p_r)
        self.progress_lines.append(Line(p_l, p_r))
        self.p = self.p + self.w/2*tan*self.d
        self.middle.add(self.p.copy())
        self.d = rotate(self.d, angle)
        self.p += self.w/2*tan*self.d
        self.add()
        
    def straight_widen(self, distance, w_new):
        w_old = self.w
        n = int(distance//self.delta)
        n = max(1, n)
        dlt = distance/n
        for i in range(n):
            t = (i+1)/n
            self.w = (1-t)*w_old + t*w_new
            self.p = self.p + dlt * self.d
            self.add()
            
    def close(self, distance, f=None):
        'Closes the route by adding straight parts and a curve'
        alpha = np.arccos(np.dot(self.d, self.d0))
        alpha *= np.sign(np.dot(ortho(self.d), self.d0))
        temp_route = Route()
        temp_route.p = self.p
        temp_route.d = self.d
        temp_route.delta = self.delta
        temp_route.curve(alpha, distance=distance, f=f)
        v = self.p0 - temp_route.p # we must achieve this vector by to straight parts!
        # linear equation:
        # a*d + b*d0 = v
        # (d | d0 ) * (a, b) = v
        (a, b) = np.linalg.inv(np.stack([self.d, self.d0], -1)) @ v
        
        # Now add the straight part, curve part, straight parth
        self.straight(a)
        self.curve(alpha, distance=distance, f=f)
        self.straight(b)
        
        self.border_l.close()
        self.border_r.close()
            
    def intersection(self, ray, plot=False):
        tl, sl = self.border_l.intersection(ray)
        tr, sr = self.border_r.intersection(ray)
        s = min_none(sl, sr)
        if plot and s:
            plt.scatter((ray.p[0]+s*ray.d[0],), (ray.p[1]+s*ray.d[1],))
        return s
        
    # Funktionen für einen schöneren Kurvenverlauf (werden in 'curve' übergeben als f)
    @staticmethod
    def f(x):
        if x<0.5:
            return 2*x
        else:
            return 2-2*x
    @staticmethod
    def g(x):
        return -x**2+x
    
        
        
def standard_route(delta=0.5):
    f = Route.f
    rt = Route()
    rt.delta = delta
    rt.curve(20*deg, 4, f)
    rt.curve(130*deg, 3, f)
    rt.curve(-110*deg, 4, f)
    rt.curve(130*deg, 5, f)
    rt.straight(3)
    rt.curve(160*deg, 6, f)
    rt.curve(-100*deg, 3, f)
    rt.straight(1.8)
    rt.curve(-60*deg, 3, f)
    rt.curve(-100*deg, 3, f)
    rt.curve(90*deg, 3, f)
    rt.curve(90*deg, 3, f)
    rt.straight(2)
    rt.curve(80*deg, 3, f)
    rt.curve_circle(-100*deg, 2)
    rt.curve_circle(+100*deg, 2)
    rt.curve_circle(+100*deg, 2)
    rt.curve_circle(-100*deg, 2)
    rt.curve_circle(+90*deg, 2)
    rt.close(1)
    return rt


    
    
# Car_v3 mit Kollisionserkennung und Resetfunktion
class Car_v3:
    def __init__(self):
        self.l = 0.3
        self.w = 0.15
        self.p = np.array([0.,0.])
        self.d = np.array([1.,0.])
        self.color = 'r'
        self.drag_heck = True # drag_heck bedeutet, das Heck wird gezogen.
        self.compute_corners()
        
    def compute_corners(self):
        if self.drag_heck:
            tf = 1/3*self.l # to front from self.p
            th = 2/3*self.l # to heck from self.p
        else:
            tf = 1/2*self.l
            th = 1/2*self.l
            
        o = ortho(self.d)
        self.corners = []
        self.corners.append(self.p - self.w/2*o + tf*self.d)
        self.corners.append(self.p + self.w/2*o + tf*self.d)
        self.corners.append(self.p + self.w/2*o - th*self.d)
        self.corners.append(self.p - self.w/2*o - th*self.d)
        
    def step(self, angle, v, delt):
        self.corners_old = self.corners
        d = rotate(self.d, angle) # Moving vector of the front
        rear_abs = self.p - self.d*self.l/3
        self.p += d * v * delt
        self.d = normed(self.p-rear_abs) if self.drag_heck else d
        self.compute_corners()
        
    def check_collision(self, pl):
        # Check the collision of the corners with a polyline or line
        smin = 9999
        for i in range(4):
            r = Ray(self.corners_old[i], self.corners[i]-self.corners_old[i])
            #r.plot() # Toggle for flying carpet effect!!
            _, s = pl.intersection(r)
            if s:
                smin = min(smin, s)
        if 0 < smin <= 1:
            return smin
        else:
            return False
    
    def reset_to_line(self, line):
        norms = [ norm(self.p-p) for p in line.pts ]
        i = np.argmin(norms)
        self.p = line.pts[i].copy()
        if i == len(line.pts)-1:
            i -= 1
        self.d = normed(normed(line.pts[i+1]-line.pts[i])) #.copy() # copy notwendig?
        self.compute_corners()
        
    def plot(self):
        ###o = ortho(self.d)
        p = patches.Polygon(self.corners, facecolor=self.color, edgecolor='black', zorder=100)
        plt.gca().add_patch(p)
        
        
class Car_v5(Car_v3):
    
    def compute_corners(self):
        super().compute_corners()
        self.edges = []
        for i in range(3):
            for t in np.arange(0., 1., 0.1):
                self.edges.append( t * self.corners[i] + (1-t) * self.corners[i+1] )
        self.edges = np.array(self.edges)
                

    def check_collision(self, pl):
        # Check the collision of the edges with a polyline or line
        
        # Schnellere Kollision
        if type(pl) in [Polyline, Polyline_thick] and False: ##
            # (1) Aus Punkten rays erstellen
            edgerays_p = self.edges_old
            edgerays_d = self.edges - self.edges_old

            # (2) s errechnen (Schnitt mit Polylines)
            ss = pl.intersection_optimized(edgerays_p, edgerays_d)
            smin = min(ss)
            if 0 < smin <= 1:
                return smin
            else:
                return False
        
        else:
            
            smin = 9999
            for i in range(len(self.edges)):
                r = Ray(self.edges_old[i], self.edges[i]-self.edges_old[i])
                _, s = pl.intersection(r)
                if s:
                    smin = min(smin, s)
            if 0 < smin <= 1:
                return smin
            else:
                return False
                       
    def step(self, angle, v, delt):
        self.corners_old = self.corners
        self.edges_old = self.edges
        d = rotate(self.d, angle) # Moving vector of the front
        rear_abs = self.p - self.d*self.l/3
        self.p += d * v * delt
        self.d = normed(self.p-rear_abs) if self.drag_heck else d
        self.compute_corners()
        
        
        
        
from copy import deepcopy


'''Definition eines Environments
Die Funktionen sind stark angelehnt an die OpenAI-Gym Environments. Durch step wird ein Befehl übergeben, die Rückgabe ist der Zustand, d. h. die Entfernung der Detektoren zum Bahnrand.'''


class Environment:
    def __init__(self, route, car, maxsteps=None):
        self.route = route
        self.car0 = car
        self.ray_angles = [+45*deg, +22*deg, 0*deg, -22*deg, -45*deg]
        self.delt = 0.1
        self.s_max_abs = 0.2
        self.velocity = 1.
        self.ray_f = 2
        self.maxsteps = maxsteps
        self.reset()
        
    def reset(self):
        self.car = deepcopy(self.car0)
        self.count_collisions = 0
        self.steps = 0
        self.rays = []
        self.rays_s = []
        self.reset_car_next = False
        self.progress = 0
        return self.step(0.)[0] # do one step to get observations
        
    def step(self, s):
        self.steps += 1
        reward = 1.
        done = False
        if self.maxsteps and self.steps > self.maxsteps:
            done = True
            
        pold, dold = 1*self.car.p, 1*self.car.d
            
        # reset to middle line?
        if self.reset_car_next:
            self.car.reset_to_line(self.route.middle)
            self.reset_car_next = False
            
        # Car does a step
        s = self.s_max_abs * np.clip(s, -1, 1)
        self.car.step(s, self.velocity, self.delt)
            
        # Check collision, maybe reset
        t_l = self.car.check_collision(self.route.border_l)
        t_r = self.car.check_collision(self.route.border_r)
        if t_l or t_r:
            t = t_l if t_l else t_r
            self.reset_car_next = True
            self.count_collisions += 1
            reward = -1.
            # put back to the colliding position
            self.car.p = (1-t)*pold + t*self.car.p
            self.car.d = (1-t)*dold + t*self.car.d
            self.car.compute_corners()
            
            self.car.p = pold
            self.car.d = dold
            self.car.step(s, self.velocity, t*self.delt) # stimmt diese zeile? muss die nicht weg? velocity einen ganz kleine faktor?
            
        # progress
        for n in range(len(self.route.progress_lines)-1, 0, -1):
            line = self.route.progress_lines[n]
            if self.car.check_collision(line):
                self.progress = max(self.progress, n) # probably not nescessary to take maximum but its safe
                break
            
        # Rays
        self.rays_s = []
        self.rays = np.array([ Ray(self.car.p, self.ray_f * rotate(self.car.d, a)) for a in self.ray_angles ])
        for r in self.rays:
            t, sl = self.route.border_l.intersection(r)
            t, sr = self.route.border_r.intersection(r)
            s = min_none(sl, sr)
            s = min_none(s, 1.)
            self.rays_s.append(s)
        self.rays_s = np.asarray(self.rays_s)
        self.ray_markers = np.array([ r.p + s*r.d for r, s, in zip(self.rays, self.rays_s) ])
        return self.rays_s, reward, done, 0
    
    def render(self, ticks=False, grid=False, lw_factor=1.):
        self.route.plot(lw_factor=lw_factor)
        self.car.plot()
        for r, s in zip(self.rays, self.rays_s):
            plt.plot([r.p[0], r.p[0]+s*r.d[0]], [r.p[1], r.p[1]+s*r.d[1]], '-x', c='gray', lw=lw_factor*1.5)
        plt.xlim([self.car.p[0]-1,self.car.p[0]+1])
        plt.ylim([self.car.p[1]-1,self.car.p[1]+1])
        if not ticks:
            plt.xticks([])
            plt.yticks([])
        if grid:
            plt.grid()
        plt.title('Number of collisions: ' + str(self.count_collisions) + ', Steps: '+str(self.steps))
        
        
class Environment_acc(Environment):
    def __init__(self, route, car, maxsteps=None):
        self.velocity_min = 1.
        self.velocity_max = 2.
        self.acceleration = 1.5
        self.progress = 0. # how far the car has gone
        super().__init__(route, car, maxsteps)
        
    def reset(self):
        self.velocity = self.velocity_min
        self.progress = 0. # how far the car has gone
        return super().reset()
        
    def step(self, s, acc=0.):
        self.steps += 1
        done = False
        if self.maxsteps and self.steps > self.maxsteps:
            done = True
            
        # Car does a step
        a = self.acceleration * np.clip(acc, -1, 1)
        self.velocity = np.clip(self.velocity + self.delt*a, self.velocity_min, self.velocity_max)
        s = self.s_max_abs * np.clip(s, -1, 1)
        self.car.step(s, self.velocity, self.delt)
        self.progress += self.velocity*self.delt
        reward = self.velocity # 1.
        
        # Check collision, maybe reset
        if self.car.check_collision(self.route.border_l) or self.car.check_collision(self.route.border_r):
            self.velocity = self.velocity_min
            self.car.reset_to_line(self.route.middle)
            self.count_collisions += 1
            reward = -1.
            
        # Rays
        self.rays_s = []
        self.rays = np.array([ Ray(self.car.p, self.ray_f * rotate(self.car.d, a)) for a in self.ray_angles ])
        for r in self.rays:
            t, sl = self.route.border_l.intersection(r)
            t, sr = self.route.border_r.intersection(r)
            s = min_none(sl, sr)
            s = min_none(s, 1.)
            self.rays_s.append(s)
        state = np.asarray(self.rays_s + [self.velocity])
        self.rays_s = np.asarray(self.rays_s)
        self.ray_markers = np.array([ r.p + s*r.d for r, s, in zip(self.rays, self.rays_s) ])
        
        return state, reward, done, 0
    
    
    

def route_video():
    rt = Route(start_p=np.array([-4.,4.]), start_angle=-90*deg, start_width=1.2, border_th=BORDER_THICKNESS)
    # Fahrbahn soll am Anfang geschlossen sein
    rt.border_l.pts = [rt.border_r.pts[0]-np.array([0,+1.]), rt.border_r.pts[0], rt.border_l.pts[0]]
    
    rt.delta = 0.05
    rt.straight(2.6)

    # Erste Kurven
    rt.curve_edgy(+22*deg)
    rt.straight(1.5)
    rt.curve_edgy(-22*deg)
    rt.straight(1.3)
    rt.curve_edgy(-23.5*deg)
    rt.straight(1.4)
    rt.curve_edgy(+32*deg)
    rt.straight(0.85)
    rt.curve_edgy(+18*deg)
    rt.straight_widen(0.7, 1.1)
    rt.curve_edgy(+12*deg)
    rt.straight_widen(0.4, 1.0)
    rt.curve_edgy(+12*deg)
    rt.straight_widen(0.4, 0.9)
    rt.curve_edgy(+39.5*deg)
    rt.straight(3)

    # 90° nach links
    rt.curve_edgy(+25*deg)
    rt.straight_widen(0.3, 0.8)
    rt.curve_edgy(+20*deg)
    rt.straight_widen(0.3, 0.8)
    rt.curve_edgy(+25*deg)
    rt.straight_widen(0.7, 0.7)
    rt.curve_edgy(+15*deg)
    rt.straight_widen(0.5, 0.6)
    rt.straight(0.3)
    rt.curve_edgy(+10*deg)
    rt.straight(0.475)

    # Das schmale C
    rt.curve_edgy(+20*deg)
    rt.straight(0.4)
    rt.curve_edgy(+20*deg)
    rt.straight_widen(0.4, 0.55)
    rt.curve_edgy(+25*deg)
    rt.straight(0.4)
    rt.curve_edgy(+20*deg)
    rt.straight(1.0)
    # rechts
    rt.curve_edgy(-20*deg)
    rt.straight(0.40)
    rt.curve_edgy(-23*deg)
    rt.straight(0.375)
    rt.curve_edgy(-24*deg)
    rt.straight(0.35)
    rt.curve_edgy(-24*deg)
    rt.straight(1.3)
    # rechts nochmal
    rt.curve_edgy(-20*deg)
    rt.straight(0.40)
    rt.curve_edgy(-22*deg)
    rt.straight(0.375)
    rt.curve_edgy(-22*deg)
    rt.straight(0.35)
    rt.curve_edgy(-22*deg)
    rt.straight_widen(0.75, 0.5)
    rt.straight(0.23)
    # links
    rt.curve_edgy(+18*deg)
    rt.straight(0.40)
    rt.curve_edgy(+20*deg)
    rt.straight(0.35)
    rt.curve_edgy(+24*deg)
    rt.straight(0.35)
    rt.curve_edgy(+25*deg)
    rt.straight(1.8)

    # letztes leichte Kurve
    rt.curve_edgy(+15*deg)
    rt.straight(0.8)
    rt.curve_edgy(+15*deg)
    rt.straight(0.8)
    rt.curve_edgy(+11*deg)
    rt.straight(0.8)
    rt.curve_edgy(+18*deg)
    rt.straight(0.8)
    rt.curve_edgy(+18*deg)
    rt.straight(0.8)
    rt.curve_edgy(+12*deg)
    rt.straight(0.95)

    # Der Trichter
    rt.straight_widen(1.9, 0.3)
    rt.straight_widen(2.1, 0.6)
    ## line_start = Line(rt.border_l.pts[0], rt.border_r.pts[1])
    
    rt.border_l.close() # borders are polygons which are created now
    rt.border_r.close()
    
    rt.c = 'grey'
    return rt

def car_video():
    #car = Car_v3()
    car = Car_v5()
    car.l *= 0.8
    car.w *= 0.8
    car.p = np.array([-4, 3.5])
    car.drag_heck = False
    return car



class EnvironmentVid(Environment):
    
    def __init__(self):
        Environment.__init__(self, route_video(), car_video())
        self.maxsteps = 9999999
    
    def render(self, ticks=False, grid=False, lw_factor=2.):
        p1 = self.route.border_l.pts[0]
        p2 = self.route.border_r.pts[0]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=self.route.c, lw=1.5*lw_factor)
        self.route.plot(lw_factor=lw_factor)
        self.car.plot()
        for r, s in zip(self.rays, self.rays_s):
            plt.scatter(r.p[0]+s*r.d[0], r.p[1]+s*r.d[1], marker='x', c='grey', s=100)
        plt.xlim([self.car.p[0]-1,self.car.p[0]+1])
        plt.ylim([self.car.p[1]-1,self.car.p[1]+1])
        if not ticks:
            plt.xticks([])
            plt.yticks([])
        if grid:
            plt.grid()
        plt.title('Number of collisions: ' + str(self.count_collisions) + ', Steps: '+str(self.steps))
        
        ###n = self.progress
        ###for n in self.progresses:
        ###    self.route.progress_lines[n].plot()