import argparse
import numpy as np
import plotly
import plotly.figure_factory as ff
from skimage import measure
from knnsearch import knnsearch


def createGrid(points, resolution=64):
    """
    constructs a 3D grid containing the point cloud
    each grid point will store the implicit function value
    Args:
        points: 3D points of the point cloud
        resolution: grid resolution i.e., grid will be NxNxN where N=resolution
                    set N=16 for quick debugging, use *N=64* for reporting results
    Returns: 
        X,Y,Z coordinates of grid vertices     
        max and min dimensions of the bounding box of the point cloud                 
    """
    max_dimensions = np.max(points,axis=0) # largest x, largest y, largest z coordinates among all surface points
    min_dimensions = np.min(points,axis=0) # smallest x, smallest y, smallest z coordinates among all surface points    
    bounding_box_dimensions = max_dimensions - min_dimensions # com6pute the bounding box dimensions of the point cloud
    max_dimensions = max_dimensions + bounding_box_dimensions/10  # extend bounding box to fit surface (if it slightly extends beyond the point cloud)
    min_dimensions = min_dimensions - bounding_box_dimensions/10
    X, Y, Z = np.meshgrid( np.linspace(min_dimensions[0], max_dimensions[0], resolution),
                           np.linspace(min_dimensions[1], max_dimensions[1], resolution),
                           np.linspace(min_dimensions[2], max_dimensions[2], resolution) )    
    
    return X, Y, Z, max_dimensions, min_dimensions

def sphere(center, R, X, Y, Z):
    """
    constructs an implicit function of a sphere sampled at grid coordinates X,Y,Z
    Args:
        center: 3D location of the sphere center
        R     : radius of the sphere
        X,Y,Z coordinates of grid vertices                      
    Returns: 
        IF    : implicit function of the sphere sampled at the grid points
    """    
    IF = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2 - R ** 2 
    return IF

def showMeshReconstruction(IF):
    """
    calls marching cubes on the input implicit function sampled in the 3D grid
    and shows the reconstruction mesh
    Args:
        IF    : implicit function sampled at the grid points
    """    
    verts, simplices, normals, values = measure.marching_cubes(IF, 0)        
    x, y, z = zip(*verts)
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    fig = ff.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=simplices,
                            title="Isosurface")
    plotly.offline.plot(fig)

def mlsReconstruction(points, normals, X, Y, Z):
    """
    surface reconstruction with an implicit function f(x,y,z) representing
    MLS distance to the tangent plane of the input surface points 
    The method shows reconstructed mesh
    Args:
        input: filename of a point cloud    
    Returns:
        IF    : implicit function sampled at the grid points
    """

    # idx stores the index to the nearest surface point for each grid point in Q.
    # we use provided knnsearch function        
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    R = points
    K = 20
    idx = knnsearch(Q, R, K)

    ################################################
    # <================START CODE<================>
    ################################################
     
    # compute the indices of the closest surface points for each point in the cloud
    csp = points[knnsearch(points, points, 1).reshape(-1)]        
    cloud_dist = np.sqrt(np.sum((points - csp) ** 2, 1))
    inv_beta = 1 / (2 * np.average(cloud_dist))

    pj, nj = points[idx], normals[idx]

    # reassign axes for easier broadcast operations 
    pj = np.swapaxes(pj, 0, 1)
    nj = np.swapaxes(nj, 0, 1)

    grid_dist = Q - pj
    phi = np.exp(-(inv_beta * np.sum(grid_dist, 2)) ** 2)
    di = np.sum(np.multiply(grid_dist, nj), 2)

    wdistsum = np.sum(np.multiply(di, phi), 0)
    wsum = np.sum(phi, 0)

    IF = (wdistsum / wsum).reshape((X.shape[0], X.shape[1], X.shape[2]))
    
    ################################################
    # <================END CODE<================>
    ################################################

    return IF 


def naiveReconstruction(points, normals, X, Y, Z):
    """
    surface reconstruction with an implicit function f(x,y,z) representing
    signed distance to the tangent plane of the surface point nearest to each 
    point (x,y,z)
    Args:
        input: filename of a point cloud    
    Returns:
        IF    : implicit function sampled at the grid points
    """

    # idx stores the index to the nearest surface point for each grid point.
    # we use provided knnsearch function
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    R = points
    K = 1
    idx = knnsearch(Q, R, K)

    ################################################
    # <================START CODE<================>
    ################################################

    idx_map = idx.reshape(-1)
    pj, nj = points[idx_map], normals[idx_map]

    tmp = np.multiply(Q - pj, nj)

    IF = np.sum(tmp, 1).reshape((X.shape[0], X.shape[1], X.shape[2]))

    print(IF.shape)

    ################################################
    # <================END CODE<================>
    ################################################

    return IF


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Basic surface reconstruction')
    parser.add_argument('--file', type=str, default = "bunny-500.pts", help='input point cloud filename')
    parser.add_argument('--method', type=str, default = "sphere",\
                        help='method to use: mls (Moving Least Squares), naive (naive reconstruction), sphere (just shows a sphere)')
    args = parser.parse_args()


    print("loading point cloud")
    #load the point cloud
    data = np.loadtxt(args.file)
    points = data[:,:3]
    normals = data[:,3:]

    print("creating grid")
    # create grid whose vertices will be used to sample the implicit function

    # uncomment this for report and presenting results 
    X,Y,Z,max_dimensions,min_dimensions = createGrid(points, 64)

    # quick debug 
    # X,Y,Z,max_dimensions,min_dimensions = createGrid(points, 16)

    print("starting reconstruction")
    if args.method == 'mls':
        print(f'Running Moving Least Squares reconstruction on {args.file}')
        IF = mlsReconstruction(points, normals, X, Y, Z)
    elif args.method == 'naive':
        print(f'Running naive reconstruction on {args.file}')
        IF = naiveReconstruction(points, normals, X, Y, Z)
    else:
        # toy implicit function of a sphere - replace this code with the correct
        # implicit function based on your input point cloud!!!
        print(f'Replacing point cloud {args.file} with a sphere!')
        center =  (max_dimensions + min_dimensions) / 2
        R = max( max_dimensions - min_dimensions ) / 4
        IF =  sphere(center, R, X, Y, Z)

    print("done")

    showMeshReconstruction(IF)