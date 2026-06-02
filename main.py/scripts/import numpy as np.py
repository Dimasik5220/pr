import numpy as np
import cv2, os, time
from collections import deque

print("="*70+"\nКУРСОВАЯ: ОПТИЧЕСКИЙ ПОТОК И ДВИЖЕНИЕ\n"+"="*70)

VIDEO_PAPKA = "D:/optical_flow_project/video"
os.makedirs(VIDEO_PAPKA, exist_ok=True)

def naiti_video():
    spisok = [f for f in os.listdir(VIDEO_PAPKA) if f.endswith(('.avi','.mp4','.mov','.mkv'))]
    if not spisok: print("Ошибка! Нет видео"); exit()
    print(f"Видео: {spisok[0]}")
    return os.path.join(VIDEO_PAPKA, spisok[0])


# 1. LUCAS-KANADE

def lukas_kanade(cap, w, h):
    print("\n--- Lucas-Kanade ---")
    back = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=36, detectShadows=False)
    trasy, pred_centry = {}, {}
    next_id = frame_num = fps_cnt = 0
    fps_start, fps_txt = time.time(), "FPS: 0"

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1

        mask = back.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        centri = []
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) > 500:
                M = cv2.moments(c)
                if M["m00"]:
                    centri.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))

        obiekty, novye = [], {}
        for cent in centri:
            mid, md = None, 100
            for oid, pc in pred_centry.items():
                d = np.hypot(pc[0]-cent[0], pc[1]-cent[1])
                if d < md: md, mid = d, oid
            if mid is None: mid, next_id = next_id, next_id+1; trasy[mid] = deque(maxlen=30)
            trasy[mid].append(cent); novye[mid] = trasy[mid]; obiekty.append((mid, cent))

        out = frame.copy()
        for oid, cent in obiekty:
            pts = list(novye[oid])
            for i in range(1, len(pts)):
                if pts[i-1] and pts[i]: cv2.line(out, pts[i-1], pts[i], (0,255,0), 2)
            cv2.circle(out, cent, 8, (0,0,255), -1)
            cv2.circle(out, cent, 8, (255,255,255), 1)
            if len(pts)>=2: cv2.arrowedLine(out, pts[-2], cent, (0,255,255), 2, tipLength=0.3)
            cv2.putText(out, str(oid), (cent[0]-10, cent[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        trasy, pred_centry = novye, {oid: cent for oid, cent in obiekty}
        fps_cnt += 1
        if time.time()-fps_start >= 1: fps_txt, fps_cnt, fps_start = f"FPS: {fps_cnt}", 0, time.time()
        cv2.putText(out, f"LUCAS | Obj:{len(obiekty)} | Fr:{frame_num}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        cv2.putText(out, fps_txt, (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.imshow("Lucas-Kanade", out)
        if cv2.waitKey(30)&0xFF==27: break
        elif cv2.waitKey(30)&0xFF==ord('s'): cv2.imwrite(f"lukas_{frame_num}.png", out); print(f"Скрин: lukas_{frame_num}.png")
    print(f"Lucas-Kanade: {frame_num} кадров"); cv2.destroyAllWindows()


# 2. FARNEBACK

def farneback(cap, w, h):
    print("\n--- Farneback ---")
    params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    ret, old = cap.read()
    if not ret: return
    old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    frame_num = fps_cnt = 0
    fps_start, fps_txt = time.time(), "FPS: 0"

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, **params)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros((h,w,3), np.uint8)
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        combined = np.vstack([frame, flow_img])
        cv2.line(combined, (0,h), (combined.shape[1],h), (255,255,255),2)
        fps_cnt += 1
        if time.time()-fps_start >= 1: fps_txt, fps_cnt, fps_start = f"FPS: {fps_cnt}", 0, time.time()
        cv2.putText(combined, "ORIGINAL", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(combined, "FARNEBACK (color=direction)", (10,h+30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(combined, fps_txt, (combined.shape[1]-120,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.putText(combined, f"Frame:{frame_num}", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
        cv2.imshow("Farneback", combined)
        old_gray = gray
        if cv2.waitKey(30)&0xFF==27: break
        elif cv2.waitKey(30)&0xFF==ord('s'): cv2.imwrite(f"farneback_{frame_num}.png", combined); print(f"Скрин: farneback_{frame_num}.png")
    print(f"Farneback: {frame_num} кадров"); cv2.destroyAllWindows()


# 3. ABSDIFF

def absdiff(cap, w, h):
    print("\n--- ABSDIFF ---")
    ret, frame1 = cap.read()
    if not ret: return
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_num = 0
    while True:
        ret, frame2 = cap.read()
        if not ret: break
        frame_num += 1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        out = frame2.copy()
        n = 0
        for c in cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) > 500:
                n += 1
                x,y,w1,h1 = cv2.boundingRect(c)
                cv2.rectangle(out, (x,y), (x+w1,y+h1), (0,0,255), 2)
        combined = np.vstack([frame2, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), out])
        cv2.line(combined, (0,h), (combined.shape[1],h), (255,255,255),2)
        cv2.line(combined, (0,2*h), (combined.shape[1],2*h), (255,255,255),2)
        status = "MOTION" if n>0 else "STATIC"
        cv2.putText(combined, f"ABSDIFF - {status} | Obj:{n} | Fr:{frame_num}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255) if n>0 else (0,255,0),2)
        cv2.imshow("ABSDIFF", combined)
        gray1 = gray2
        if cv2.waitKey(30)&0xFF==27: break
    print(f"ABSDIFF: {frame_num} кадров"); cv2.destroyAllWindows()


# 4. MOG2 + ПОДСЧЁТ УНИКАЛЬНЫХ ОБЪЕКТОВ

def mog2(cap, w, h):
    print("\n--- MOG2 с подсчётом ---")
    back = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    pred_centry, trasy = {}, {}
    frame_num = next_id = max_odn = 0
    vse_id = set()
    MIN_PL, MAX_PL, BLIZKO = 400, 5000, 50

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1
        mask = back.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.medianBlur(mask, 5)
        obekty = []
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if MIN_PL <= cv2.contourArea(c) <= MAX_PL:
                x,y,wb,hb = cv2.boundingRect(c)
                if 0.2 <= wb/hb <= 5:
                    cent = (x+wb//2, y+hb//2)
                    mid, md = None, 100
                    for oid, pc in pred_centry.items():
                        d = np.hypot(pc[0]-cent[0], pc[1]-cent[1])
                        if d < md: md, mid = d, oid
                    if mid is None: mid, next_id = next_id, next_id+1; trasy[mid] = deque(maxlen=30); vse_id.add(mid)
                    trasy[mid].append(cent); obekty.append((mid, cent, (x,y,wb,hb), trasy[mid]))
        max_odn = max(max_odn, len(obekty))
        out = frame.copy()
        for oid, cent, bbox, traj in obekty:
            x,y,wb,hb = bbox
            cv2.rectangle(out, (x,y), (x+wb,y+hb), (0,255,0), 2)
            cv2.circle(out, cent, 5, (0,0,255), -1)
            pts = list(traj)
            for i in range(1,len(pts)):
                if pts[i-1] and pts[i]: cv2.line(out, pts[i-1], pts[i], (0,255,255), 2)
            if oid in pred_centry:
                pc = pred_centry[oid]
                if abs(cent[0]-pc[0])>5 or abs(cent[1]-pc[1])>5:
                    cv2.arrowedLine(out, pc, cent, (255,0,0), 2, tipLength=0.3)
            cv2.putText(out, str(oid), (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        pred_centry = {oid: cent for oid, cent, _, _ in obekty}
        cv2.putText(out, f"Current:{len(obekty)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        cv2.putText(out, f"TOTAL:{len(vse_id)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(out, f"Max at once:{max_odn}", (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        cv2.putText(out, f"Frame:{frame_num}", (10,120), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
        cv2.imshow("MOG2", out)
        if cv2.waitKey(30)&0xFF==27: break
        elif cv2.waitKey(30)&0xFF==ord('s'): cv2.imwrite(f"mog2_{frame_num}.png", out); print(f"Скрин: mog2_{frame_num}.png")
    print(f"\n{'='*60}\nИТОГИ:\n  Всего объектов: {len(vse_id)}\n  Максимум одновременно: {max_odn}\n  Кадров: {frame_num}\n{'='*60}")
    cv2.destroyAllWindows()


# 5. СРАВНЕНИЕ

def sravni(video_path):
    print("\n--- Сравнение скорости ---")
    cap = cv2.VideoCapture(video_path)
    back = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=36, detectShadows=False)
    start, frames = time.time(), 0
    while frames < 100:
        ret, frame = cap.read()
        if not ret: break
        back.apply(frame); frames += 1
    lk_fps = frames/(time.time()-start)
    cap.release()
    cap = cv2.VideoCapture(video_path)
    ret, old = cap.read()
    old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    start, frames = time.time(), 0
    while frames < 100:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.calcOpticalFlowFarneback(old_gray, gray, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        old_gray, frames = gray, frames+1
    far_fps = frames/(time.time()-start)
    cap.release()
    print(f"\n{'Lucas-Kanade':<20} {lk_fps:.1f} FPS")
    print(f"{'Farneback':<20} {far_fps:.1f} FPS")
    if far_fps>0: print(f"Lucas-Kanade быстрее в {lk_fps/far_fps:.1f} раз")

# ГЛАВНОЕ МЕНЮ

def main():
    video_path = naiti_video()
    cap_info = cv2.VideoCapture(video_path)
    if not cap_info.isOpened(): print("Ошибка"); return
    w, h = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_info.release()
    print(f"\nВидео: {os.path.basename(video_path)} ({w}x{h})")
    while True:
        print("\n"+"="*60+"\n1.Lucas-Kanade 2.Farneback 3.ABSDIFF 4.MOG2 5.Сравнить 6.Всё 0.Выход\n"+"="*60)
        vibor = input("Твой выбор: ").strip()
        if vibor=="1": cap=cv2.VideoCapture(video_path); lukas_kanade(cap,w,h); cap.release()
        elif vibor=="2": cap=cv2.VideoCapture(video_path); farneback(cap,w,h); cap.release()
        elif vibor=="3": cap=cv2.VideoCapture(video_path); absdiff(cap,w,h); cap.release()
        elif vibor=="4": cap=cv2.VideoCapture(video_path); mog2(cap,w,h); cap.release()
        elif vibor=="5": sravni(video_path)
        elif vibor=="6":
            for mode in [lukas_kanade, farneback, absdiff, mog2]:
                cap=cv2.VideoCapture(video_path); mode(cap,w,h); cap.release(); input("\nНажми Enter...")
            sravni(video_path)
        elif vibor=="0": print("До свидания!"); break
        else: print("Не понял")

if __name__ == "__main__":
    main()