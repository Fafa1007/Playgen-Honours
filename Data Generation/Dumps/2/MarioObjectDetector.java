package engine.core;

import engine.core.MarioLevel;

public class MarioObjectDetector {

    private MarioLevel level;
    private int screenWidth, screenHeight;

    public MarioObjectDetector(MarioLevel level, int screenWidth, int screenHeight) {
        this.level = level;
        this.screenWidth = screenWidth;
        this.screenHeight = screenHeight;
    }

    /**
     * Detect objects currently visible on screen
     * @param cameraX current camera x in pixels
     * @param cameraY current camera y in pixels
     * @return boolean array: [pipeOnScreen, coinOnScreen, flagOnScreen]
     */
    public boolean[] detectObjects(int cameraX, int cameraY) {
        boolean pipe = false;
        boolean coin = false;
        boolean flag = false;

        int startTileX = cameraX / 16;
        int startTileY = cameraY / 16;
        int endTileX = Math.min(level.tileWidth, (cameraX + screenWidth) / 16 + 1);
        int endTileY = Math.min(level.tileHeight, (cameraY + screenHeight) / 16 + 1);

        for (int x = startTileX; x < endTileX; x++) {
            for (int y = startTileY; y < endTileY; y++) {
                int block = level.getBlock(x, y);
                switch (block) {
                    case 7:  // coin block
                    case 11: // question coin block
                    case 15: // coin
                    case 49: // invisible coin block
                        coin = true;
                        break;
                    case 18: case 19: case 20: case 21: case 52: // pipes
                        pipe = true;
                        break;
                    case 39: case 40: // flag pole
                        flag = true;
                        break;
                }
            }
        }

        return new boolean[]{pipe, coin, flag};
    }
}
