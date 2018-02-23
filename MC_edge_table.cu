/* This file is part of the Marching Cubes GPU based algorithm based on 
 * Paul Bourke's tabulation approach to marching cubes
 * http://paulbourke.net/geometry/polygonise/
 *
 *
 * We model cubes with 8 vertices labelled as below
 *
 *
 *            4--------(4)---------5
 *           /|                   /|
 *          / |                  / |
 *         /  |                 /  |
 *       (7)  |               (5)  |
 *       /    |               /    |
 *      /    (8)             /    (9)
 *     /      |             /      |
 *    7---------(6)--------6       |
      |       |            |       |
 *    |       0------(0)---|-------1
 *    |      /             |      /
 *   (11)   /             (10)   /
 *    |    /               |    /
 *    |  (3)               |  (1)
 *    |  /                 |  /
 *    | /                  | /
 *    |/                   |/
 *    3---------(2)--------2
 *
 * where X axis is horizontal, +ve to right
 *       Y axis is vertical, +ve upwards
 *       Z axis is into page, +ve towards back
 *
 * 0: ( x,   y,   z+1 )  4: ( x,   y+1,   z+1 )
 * 1: ( x+1, y,   z+1 )  5: ( x+1, y+1,   z+1 )
 * 2: ( x+1, y,   z   )  6: ( x+1, y+1,   z   )
 * 3: ( x,   y,   z   )  7: ( x,   y+1,   z   )
 *
 * There are 12 edges, 0 - 11 where each edge connectes two vertices as follows:
 *
 * 0: 0, 1       1: 1, 2       2: 2, 3       3:  3, 0
 * 4: 4, 5       5: 5, 6       6: 6, 7       7:  7, 4
 * 8: 0, 4       9: 1, 5      10: 2, 6      11:  3, 7
 */

 // NB Below, these are ordered from lower to higher value

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;

 __constant__
uint16_t EDGE_VERTICES[12][2] = {
    { 0, 1 }, { 2, 1 }, { 3, 2 }, { 3, 0 },
    { 4, 5 }, { 6, 5 }, { 7, 6 }, { 7, 4 },
    { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } 
 };

/*
 * This file describes the relationship between the vertices under the surface
 * and the edges which are therefore impacted
 * There are 256 distinct entries
 */
__constant__
uint16_t EDGE_TABLE[256]={
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 
};