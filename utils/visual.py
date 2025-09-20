try:
    import pygame
    PYGAME_AVAILABLE = True
except:
    PYGAME_AVAILABLE = False

class Visual:
    def __init__(self, cell=30, size=10, fps=5):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame no disponible")
        pygame.init()
        pygame.mixer.init()  # inicializar mixer solo una vez
        self.cell = cell
        self.size = size
        self.fps = fps
        self.top_margin = 40
        self.screen = pygame.display.set_mode((cell*size, cell*size + self.top_margin))
        pygame.display.set_caption("Snake AI - 10x10")
        self.clock = pygame.time.Clock()
        self.apple_img = None
        self.font = pygame.font.SysFont('Arial', 20)

        # Cargar sonidos
        self.sound_start = pygame.mixer.Sound("utils/rene_hola.mp3")
        self.sound_gameover = pygame.mixer.Sound("utils/rene_miau.mp3")

        # reproducir sonido de inicio
        self.sound_start.play()

    def draw(self, env):
        import pygame

        # manejo de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and getattr(env, 'done', False):
                    env.reset()
                    env.sound_played = False

        self.screen.fill((0,0,0))

        # contador
        score_text = self.font.render(f"Contador: {env.apples_eaten} / {env.max_apples}", True, (255,255,255))
        self.screen.blit(score_text, (5,5))

        # Game Over
        if getattr(env, 'done', False):
            go_text = self.font.render("GAME OVER - Presiona R para reiniciar", True, (255,0,0))
            self.screen.blit(go_text, (50,20))
            if not getattr(env, 'sound_played', False):
                self.sound_gameover.play()
                env.sound_played = True

        # Victoria
        if getattr(env, 'apples_eaten',0) >= getattr(env, 'max_apples',35):
            win_text = self.font.render("¡FELICIDADES! La serpiente ganó 🎉", True, (0,255,0))
            self.screen.blit(win_text, (20,20))

        # cuadrícula
        for y in range(self.size):
            for x in range(self.size):
                rect = pygame.Rect(x*self.cell, y*self.cell + self.top_margin, self.cell, self.cell)
                pygame.draw.rect(self.screen,(40,40,40),rect,1)

        # manzana
        if self.apple_img is None:
            self.apple_img = pygame.image.load("utils/loco_manzana.png").convert_alpha()
            self.apple_img = pygame.transform.scale(self.apple_img,(self.cell-2,self.cell-2))
        if env.apple is not None:
            ax, ay = env.apple
            self.screen.blit(self.apple_img, (ax*self.cell +1, ay*self.cell +1 + self.top_margin))

        # serpiente
        for i,(x,y) in enumerate(env.snake):
            r = pygame.Rect(x*self.cell+1, y*self.cell+1 + self.top_margin, self.cell-2, self.cell-2)
            if i==0:
                pygame.draw.rect(self.screen,(30,200,30),r)
            else:
                pygame.draw.rect(self.screen,(80,180,80),r)

        pygame.display.flip()
        self.clock.tick(self.fps)