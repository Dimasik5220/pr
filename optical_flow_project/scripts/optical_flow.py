# =====================================================
# УЛУЧШЕННОЕ ТЕСТОВОЕ ВИДЕО (с текстурами для трекинга)
# =====================================================

def create_test_video_with_texture():
    """Создаёт видео с текстурами для лучшего трекинга точек"""
    print("\n📹 Создаю улучшенное тестовое видео с текстурами...")
    
    video_dir = 'D:/optical_flow_project/video'
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, 'test_video.avi')
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    for i in range(300):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 40, 60)
        
        # === ДОБАВЛЯЕМ ТЕКСТУРЫ ДЛЯ ТРЕКИНГА ===
        # Рисуем шахматную сетку (создаёт много угловых точек!)
        grid_size = 40
        for x in range(0, 640, grid_size):
            for y in range(0, 480, grid_size):
                if (x // grid_size + y // grid_size) % 2 == 0:
                    cv2.rectangle(frame, (x, y), (x + grid_size, y + grid_size), (80, 80, 100), -1)
        
        # Движущийся яркий объект (шар с текстурой)
        x = int(320 + 250 * np.sin(i * 0.02))
        y = int(240 + 150 * np.cos(i * 0.03))
        
        # Шар с градиентом (больше точек для трекинга)
        for r in range(30, 0, -5):
            color = (0, 255 - r * 8, r * 8)
            cv2.circle(frame, (x, y), r, color, -1)
        
        # Несколько дополнительных движущихся точек
        x2 = int(100 + 200 * np.sin(i * 0.05))
        y2 = int(350 + 50 * np.cos(i * 0.07))
        cv2.circle(frame, (x2, y2), 10, (255, 100, 0), -1)
        
        x3 = int(500 + 100 * np.sin(i * 0.08))
        y3 = int(100 + 80 * np.cos(i * 0.04))
        cv2.circle(frame, (x3, y3), 8, (100, 200, 255), -1)
        
        cv2.putText(frame, "TEST VIDEO - With Texture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Улучшенное видео создано: {video_path}")
    return video_path


# =====================================================
# УЛУЧШЕННЫЙ Lucas-Kanade (больше точек, ярче линии)
# =====================================================

# Изменённые параметры для поиска БОЛЬШЕ точек
feature_params_enhanced = dict(
    maxCorners=200,      # ← увеличено со 100 до 200
    qualityLevel=0.15,   # ← уменьшено с 0.3 (хватаем даже слабые углы)
    minDistance=5,       # ← уменьшено с 7 (точки могут быть ближе)
    blockSize=5          # ← уменьшено с 7
)

# Параметры для Lucas-Kanade (увеличенное окно для быстрых движений)
lk_params_enhanced = dict(
    winSize=(21, 21),    # ← увеличено с (15, 15)
    maxLevel=3,          # ← увеличено с 2
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

def run_lucas_kanade_enhanced(cap):
    """Улучшенная версия с яркими траекториями"""
    print("Запуск Lucas-Kanade (улучшенный)...")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    ret, old_frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр")
        return
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Используем улучшенные параметры
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params_enhanced)
    print(f"Найдено начальных точек: {len(p0) if p0 is not None else 0}")
    
    trajectories = {}
    if p0 is not None:
        for i, point in enumerate(p0):
            trajectories[i] = deque(maxlen=trajectory_length)
            trajectories[i].append(point.ravel())
    
    # Маска для рисования траекторий (яркие линии!)
    mask = np.zeros_like(old_frame)
    
    frame_count = 0
    fps_start_time = time.time()
    fps_counter = 0
    fps_text = "FPS: 0"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params_enhanced)
            
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                new_trajectories = {}
                new_p0 = []
                
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    
                    idx = list(trajectories.keys())[i] if i < len(trajectories) else i
                    if idx in trajectories:
                        trajectories[idx].append(new.ravel())
                    else:
                        trajectories[idx] = deque(maxlen=trajectory_length)
                        trajectories[idx].append(new.ravel())
                    new_trajectories[idx] = trajectories[idx]
                    new_p0.append(new)
                    
                    # === РИСУЕМ ЯРКИЕ ТРАЕКТОРИИ ===
                    points = list(trajectories[idx])
                    for j in range(1, len(points)):
                        if points[j-1] is not None and points[j] is not None:
                            pt1 = (int(points[j-1][0]), int(points[j-1][1]))
                            pt2 = (int(points[j][0]), int(points[j][1]))
                            # Градиент цвета от жёлтого к зелёному
                            color = (0, 255 - j * 8, 255)
                            cv2.line(mask, pt1, pt2, color, 3)  # ← толщина 3 вместо 2
                    
                    # Рисуем текущую точку (ярко-красная)
                    cv2.circle(frame, (int(a), int(b)), 6, (0, 0, 255), -1)
                    cv2.circle(frame, (int(a), int(b)), 6, (255, 255, 255), 1)
                
                trajectories = new_trajectories
                p0 = np.array(new_p0).reshape(-1, 1, 2) if new_p0 else np.array([])
                
                # Если точек осталось мало - ищем новые
                if len(p0) < 50:
                    p_new = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params_enhanced)
                    if p_new is not None and len(p_new) > 0:
                        if len(p0) == 0:
                            p0 = p_new
                        else:
                            p0 = np.vstack((p0, p_new))
                        
                        for i in range(len(p_new)):
                            idx = len(trajectories) + i
                            trajectories[idx] = deque(maxlen=trajectory_length)
                            trajectories[idx].append(p_new[i].ravel())
        
        # Объединяем кадр и траектории
        result = cv2.add(frame, mask)
        
        # Информация на экран
        cv2.putText(result, f"Lucas-Kanade - Active Points: {len(p0) if p0 is not None else 0}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Trajectories: {len(trajectories)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(result, f"Frame: {frame_count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps_text = f"FPS: {fps_counter}"
            fps_counter = 0
            fps_start_time = time.time()
        cv2.putText(result, fps_text, (width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Lucas-Kanade - Trajectories', result)
        
        old_gray = frame_gray.copy()
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            cv2.imwrite(f'lk_trajectories_{frame_count}.png', result)
            print(f"Скриншот сохранён: lk_trajectories_{frame_count}.png")
    
    cv2.destroyAllWindows()