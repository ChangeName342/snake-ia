try:
    import pygame
    PYGAME_AVAILABLE = True
except:
    PYGAME_AVAILABLE = False
import os

class Visual:
    def __init__(self, cell=30, size=10, fps=10):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame no disponible")
        pygame.init()
        pygame.mixer.init()  # inicializar mixer solo una vez
        self.cell = cell
        self.size = size
        self.fps = fps
        self.top_margin = 60
        self.screen = pygame.display.set_mode((cell*size, cell*size + self.top_margin))
        pygame.display.set_caption("Snake AI - 10x10")
        self.clock = pygame.time.Clock()
        self.apple_img = None
        self.font = pygame.font.SysFont('Arial', 20)

        # Cargar sonidos (si existen)
        if os.path.exists("utils/rene_hola.mp3"):
            self.sound_start = pygame.mixer.Sound("utils/rene_hola.mp3")
            self.sound_start.play()
        else:
            self.sound_start = None

        if os.path.exists("utils/rene_miau.mp3"):
            self.sound_gameover = pygame.mixer.Sound("utils/rene_miau.mp3")
        else:
            self.sound_gameover = None

        # Cargar imagen de manzana
        if os.path.exists("utils/loco_manzana.png"):
            self.apple_img = pygame.image.load("utils/loco_manzana.png").convert_alpha()
            self.apple_img = pygame.transform.scale(self.apple_img,(self.cell-2,self.cell-2))

    def draw(self, env):
        import pygame

        # Manejo de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and getattr(env, 'done', False):
                    env.reset()
                    env.sound_played = False

        # Fondo
        self.screen.fill((20,20,20))

        # Cuadrícula
        for y in range(self.size):
            for x in range(self.size):
                rect = pygame.Rect(x*self.cell, y*self.cell + self.top_margin, self.cell, self.cell)
                pygame.draw.rect(self.screen,(60,60,60),rect,1)

        # Manzana
        if env.apple is not None:
            ax, ay = env.apple
            if self.apple_img:
                self.screen.blit(self.apple_img, (ax*self.cell +1, ay*self.cell +1 + self.top_margin))
            else:
                # fallback si no hay imagen
                pygame.draw.circle(self.screen, (255,0,0), 
                    (ax*self.cell + self.cell//2, ay*self.cell + self.cell//2 + self.top_margin), self.cell//2)

        # Serpiente con gradiente
        for i,(x,y) in enumerate(env.snake):
            r = pygame.Rect(x*self.cell+1, y*self.cell+1 + self.top_margin, self.cell-2, self.cell-2)
            color_factor = 200 - int(i*150/max(len(env.snake),1))
            if i==0:
                color = (30, 200, 30)  # cabeza
            else:
                color = (30, color_factor, 30)  # cuerpo con gradiente
            pygame.draw.rect(self.screen, color, r)

        # Flecha de dirección
        if len(env.snake)>0:
            hx, hy = env.snake[0]
            dx, dy = env.direction.dx, env.direction.dy
            pygame.draw.line(self.screen,
                             (255,255,0),
                             (hx*self.cell+self.cell//2, hy*self.cell+self.cell//2 + self.top_margin),
                             (hx*self.cell+self.cell//2+dx*10, hy*self.cell+self.cell//2+dy*10 + self.top_margin), 3)

        # HUD: contador, agente, tiempo
        score_text = self.font.render(f"Manzanas: {env.apples_eaten} / {env.max_apples}", True, (255,255,255))
        self.screen.blit(score_text, (5,5))

        agent_text = self.font.render(f"Agente: {getattr(env,'agent_name','Desconocido')}", True, (255,255,0))
        self.screen.blit(agent_text, (5,25))

        time_text = self.font.render(f"Tiempo: {getattr(env,'elapsed',0):.2f}s", True, (255,255,255))
        self.screen.blit(time_text, (5,45))

        # Game Over
        if getattr(env, 'done', False):
            go_text = self.font.render("GAME OVER - Presiona R para reiniciar", True, (255,0,0))
            self.screen.blit(go_text, (50, self.top_margin//2))
            if not getattr(env, 'sound_played', False) and self.sound_gameover:
                self.sound_gameover.play()
                env.sound_played = True

        # Victoria
        if getattr(env, 'apples_eaten',0) >= getattr(env, 'max_apples',35):
            win_text = self.font.render("¡FELICIDADES! La serpiente ganó 🎉", True, (0,255,0))
            self.screen.blit(win_text, (50, self.top_margin//2))

        pygame.display.flip()
        self.clock.tick(self.fps)
