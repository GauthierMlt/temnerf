def get_params(d):
    if d == 'walnut':
        aspect_ratio = 2240./2368. 
        # angle of view
        # camera is on a circle 343 from origin at (x,y,0)
        radius = 343
        object_size = 39 # safer to overestimate
        return 1, radius, object_size, aspect_ratio
    elif d == 'jaw':
        # sRBA
        aspect_ratio = 275./331.
        # angle of view
        # camera is on a circle 163 from origin at (x,y,0)
        radius = 163
        object_size = 48 # safer to overestimate
        return 3, radius, object_size, aspect_ratio
    elif d == "au_ag":
        radius = 40
        object_size = 36

        # radius = 512
        # object_size = 128
        
        return 1, radius, object_size, 1 

    else:
        raise NotImplementedError()