def calculate_angle(p1,p2):
    p1 = p1[0]
    p2 = p2[0]
    return math.atan((20 + ((320 - p1) * 22.5) / (p2 - p1)) / 43.5)
           
 	