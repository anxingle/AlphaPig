import unittest
from ChessBoard import ChessBoard
import ChessHelper
from ChessHelper import printBoard


class ChessBoardTest(unittest.TestCase):
    def setUp(self):
        self.a = 1

    def test_putpiece1(self):
        cb = ChessBoard()

        self.assertEqual(0, cb.put_piece(0, 0, 1))
        self.assertEqual(-2, cb.put_piece(0, 0, 1))
        self.assertEqual(-1, cb.put_piece(-1, 0, 1))
        self.assertEqual(-1, cb.put_piece(0, 16, 1))

    def test_putpiece2(self):
        cb = ChessBoard()
        muser = 1
        for i in xrange(4):
            for j in xrange(15):
                return_value = cb.put_piece(i, j, muser)
                self.assertEqual(0, return_value)
                muser = 2 if muser == 1 else 1

    def test_putpiece2(self):
        cb = ChessBoard()
        muser = 1
        for i in xrange(4):
            for j in xrange(15):
                return_value = cb.put_piece(i, j, muser)
                self.assertEqual(0, return_value)
                muser = 2 if muser == 1 else 1

    def test_putpiece3(self):
        cb = ChessBoard()
        ChessHelper.playRandomGame(cb)


if __name__ == '__main__':
    unittest.main()
