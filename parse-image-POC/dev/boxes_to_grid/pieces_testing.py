from parse_images.boxes_to_grid.pieces import *


if __name__ == '__main__':
    for piece in PIECES:
        print('----------------------')
        print('name:', piece.name)
        
        for i, variant in enumerate(piece.variants):
            print('\tvariant', i+1)
            Piece._print_grid(variant, prefix='\t')