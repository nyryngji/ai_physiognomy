from django.shortcuts import render,redirect
from .forms import ImageUploadForm
import os
import cv2
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms, models
from PIL import Image
from torch import nn
from django.contrib.sessions.models import Session

def main(request):
    Session.objects.all().delete()

    dir = 'D:\\mysite\\media\\media'
    for file in os.scandir(dir):
        os.remove(file.path)
        
    context = {}
    return render(request, 'face/main_page.html', context)

def choose(request):
    context = {}
    return render(request,'face/choose_page.html', context)

def loading(request):
    context = {}
    return render(request,'face/loading.html', context)

def choose_gender(request):
    context = {}
    return render(request,'face/choose_gender_page.html', context)

def upload_page1(request):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()
            image_path = image_instance.image.path

            m = tf.keras.models.load_model('D:\\mysite\\face\\incep.h5')

            img = cv2.imread(image_path)
            img = cv2.resize(img, (150,150), interpolation=cv2.INTER_LINEAR)
            img_scaled = (img / 255)
            img_scaled = np.expand_dims(img_scaled, axis=0)

            # 모델을 사용하여 예측 수행
            predictions = m.predict(img_scaled)

            # 예측 결과 해석
            predicted_class = np.argmax(predictions, axis=1)

            # 클래스 이름 매핑 (필요한 경우)
            classes = ['비글','치와와','골든 리트리버','포메라니안','퍼그','시베리안 허스키']
            
            pred = classes[predicted_class[0]]

            request.session['dog_pred'] = str(pred)
            request.session['dog_idx'] = str(predicted_class[0])

            return redirect('result_page')
    else:
        form = ImageUploadForm()
        return render(request, 'face/file_upload_page_dog.html', {'form': form})
    
def upload_page2(request):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()
            image_path = image_instance.image.path

            class CustomModel(nn.Module):
                def __init__(self, base_model, num_classes, dropout):
                    super(CustomModel, self).__init__()
                    self.base_model = base_model
                    self.dropout = nn.Dropout(p=dropout)
                    self.fc = nn.Linear(460800, num_classes)
                
                def forward(self, x):
                    x = self.base_model.features(x)
                    x = self.dropout(x)
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)
                    return x

            # Function to load the model
            def load_model(model_path, num_classes, dropout):
                base_model = models.efficientnet_b5(weights=None)
                model = CustomModel(base_model, num_classes, dropout)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                return model

            # Function to predict the breed of the cat
            def predict_image(image, model, transform, class_names):
                image = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(image)
                    _, preds = torch.max(outputs, 1)

                return class_names[preds[0]]

            image = Image.open(image_path)

            image_size = 456
            dropout = 0.3
            data_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
            class_names = ['아비시니안', '아메리칸밥테일', '아메리칸컬', '아메리칸쇼트헤어', '뱅갈', 
                            '버만', '봄베이', '브리티시쇼트헤어', '이집션마우', '엑조틱 숏헤어', '메인쿤', 
                            '맹크스', '노르웨이숲', '페르시안', '랙돌', '러시안 블루', '스코티시폴드', 
                            '샤미즈', '스핑크스', '터키시앙고라']
            
            model_path = "D:\\mysite\\face\\cat_breeds_efficientnet_b5_83_3H.pth"
            model = load_model(model_path, len(class_names), dropout)

            # Predict breed
            prediction = predict_image(image, model, data_transform, class_names)

            request.session['cat_pred'] = prediction
            request.session['cat_idx'] = str(class_names.index(prediction))

            return redirect('result_page')
    else:
        form = ImageUploadForm()
        return render(request,'face/file_upload_page_cat.html', {'form':form})

def upload_page3(request):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()
            image_path = image_instance.image.path

            m = tf.keras.models.load_model('D:\\mysite\\face\\man94.h5')

            img = cv2.imread(image_path)
            img = cv2.resize(img, (150,150), interpolation=cv2.INTER_LINEAR)
            img_scaled = (img / 255)
            img_scaled = np.expand_dims(img_scaled, axis=0)

            # 모델을 사용하여 예측 수행
            predictions = m.predict(img_scaled)

            # 예측 결과 해석
            predicted_class = np.argmax(predictions, axis=1)

            # 클래스 이름 매핑 (필요한 경우)
            classes = ['시우민', '디오', '슈가', '탑']
            
            pred = classes[predicted_class[0]]

            request.session['image_path'] = image_path

            request.session['pred_man'] = str(pred)
            request.session['idx_man'] = str(predicted_class[0])

            return redirect('result_page')
    else:
        form = ImageUploadForm()
        return render(request, 'face/file_upload_page_man.html', {'form': form})

def upload_page4(request):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()
            image_path = image_instance.image.path

            m = tf.keras.models.load_model('D:\\mysite\\face\\girl95.h5')

            img = cv2.imread(image_path)
            img = cv2.resize(img, (150,150), interpolation=cv2.INTER_LINEAR)
            img_scaled = (img / 255)
            img_scaled = np.expand_dims(img_scaled, axis=0)

            # 모델을 사용하여 예측 수행
            predictions = m.predict(img_scaled)

            # 예측 결과 해석
            predicted_class = np.argmax(predictions, axis=1)

            # 클래스 이름 매핑 (필요한 경우)
            classes = ['제니', '카리나', '지수', '윈터']
            
            pred = classes[predicted_class[0]]

            request.session['image_path'] = image_path

            request.session['pred_girl'] = str(pred)
            request.session['idx_girl'] = str(predicted_class[0])

            return redirect('result_page')
    else:
        form = ImageUploadForm()
        return render(request, 'face/file_upload_page_dog.html', {'form': form})
    
def result(request):

    dog_pred = request.session.get('dog_pred')
    cat_pred = request.session.get('cat_pred')
    man_pred = request.session.get('pred_man')
    girl_pred = request.session.get('pred_girl')

    dog_face = ['크고 동그란 눈, 온순한 인상',
            '작은 얼굴, 큰 눈',
            '환한 미소, 부드러운 분위기',
            '귀여운 얼굴, 작은 눈',
            '주름진 얼굴, 큰 눈',
            '뚜렷한 얼굴선, 이색적인 눈동자']
    
    dog_personality = ['활발함, 똑똑함, 온화함',
                   '용감함, 빠름, 고집이 셈',
                   '온순함, 애정이 깊음, 정직함',
                   '장난기가 많음, 독립적, 애교 많음',
                   '고집 셈, 발랄함, 장난치기 좋아함',
                   '카리스마, 활발함, 은근 허당끼']

    dog_info = ['모든 면에서 매우 활동적인 강아지로,\n총명한 눈을 가진 당신과 매우 닮았소!',
            '가장 작은 체구를 가졌지만 매우 용감하오!\n위의 얼굴처럼 되지 않도록 주의하시오!',
            '친절하고 온화한 분위기를 가진 강아지로,\n주변에서 사랑을 듬뿍 받는 당신과 닮았소!',
            '남들이 부러워할 귀여운 외모를 가졌소!\n매력적인 당신과 참 잘 어울린다 생각하오.',
            '특유의 불쌍한 표정이 매력적이오!\n당신이 부탁하면 안 들어줄 사람이 없을 것이오.',
            '개X 마이웨이 성향이 강한 견종이오!\n하지만 매력적인 멋진 얼굴을 가진 당신과 닮았소!']
    
    cat_face = ['귀엽고 아름다움', '사냥꾼 눈의 야생적인 외모', '까칠하고 사나워 보이는 외모', '크고 동글동글하며 붙임성 있음', 
                '야생적인 외모와 날카로운 턱선', '둥근 머리와 파란 눈을 가진 외모', '코가 짧고 약간 뭉툭하며, 전체적으로 둥글함', 
                '둥글고 큰 얼굴에 넓은 눈과 두꺼운 볼살', '귀가 크고 뾰족함, 중간 크기의 머리와 큰 눈', '납작한 코와 큰 눈, 둥글고 넓은 얼굴', 
                '큰 머리와 뚜렷한 이목구비', '둥근 머리와 큰 눈을 가진 외모', '삼각형 얼굴형과 큰 눈, 전체적으로 강한 인상', 
                '얼굴이 넓고 둥글며, 귀가 작고 둥글게 말려있음', '중간 크기의 머리와 큰 파란 눈', '얼굴이 날렵하고 귀가 크고 뾰족함', 
                '코가 짧고, 얼굴 전체가 둥근 인상', '얼굴이 삼각형 모양이고 코가 길고 뾰족함', '얼굴이 삼각형 모양이고 눈이 크고 주름짐', 
                '코가 길고 날렵하며, 귀가 크고 뾰족함']
        
    cat_personality = ['예민함, 호기심 많음', '태평함, 적응력 MAX', '차분함, 영리함', '낙천적, 쾌활함', 
                    '활발함, 호기심 많음', '온순함, 애정 많음', '낙천적, 애교 많음', '태평함, 차분함', 
                    '활기참, 영리함', '온순함, 낙천적', '친근함, 쾌활함', '활발함, 애정 많음', 
                    '독립적, 차분함', '차분함, 태평함', '온순함, 낙천적', '차분함, 영리함', 
                    '낙천적, 애교 많음', '활기참, 영리함', '활발함, 애정 많음', '영리함, 활기참']
    
    cat_info = ['무엇보다 호기심이 왕성하고 똑똑한 고양이로,\n 귀엽고 아름다운게 당신과 닮았지만 성격은 장점만 닮은걸로?',
            '영리하고 활동적이며 다정한 고양이로,\n 강인한 눈이 당신을 닮았소! 성격도 닮았으면 좋겠소!',
            '호기심이 많고, 동료애가 있으며, 매우 사람을 잘 따르는 고양이로,\n 야생미넘치는 외모가 당신을 닮았소!',
            '온화하고 점잖으면서도 애정이 많은 고양이로,\n 둥근 얼굴이 당신과 닮았소! 혹시 성격도?',
            '활기차고 장난을 많이 치며 주변을 탐험하는 것을 좋아하는 고양이로,\n 강인한 외모가 당신과 닮았소!',
            '사람에게 잘 따르고 조용히 곁에 머무르는 것을 좋아하는 고양이로,\n 안정감을 주는 외모가 당신과 비슷하오!',
            '가족과 함께 놀고 시간을 보내는 것을 즐기는 고양이로,\n 시크한 외모가 당신을 표현한다!',
            '혼자서 조용히 지내는 것을 좋아하며 독립적인 고양이로,\n 귀엽게 생긴 당신의 외모가 부럽소!',
            '빠르고 민첩하게 움직이며 사냥 놀이를 즐기는 고양이로,\n 역삼각인 당신의 외모가 부러울수도?',
            '차분하고 느긋하게 집안을 돌아다니며 휴식을 즐기는 고양이로,\n 도도하면서도 귀여운 얼굴이 당신을 닮았소!',
            '사람들과 잘 어울리고 다양한 활동을 즐기는 고양이로,\n 뚜렷한 이목구비가 당신과 닮았다! 잘생긴 당신이 부럽다!',
            '활발하게 뛰어다니며 놀이와 탐험을 좋아하는 고양이로,\n 무뚝뚝하면서도 순수한 외모가 당신을 닮았소!',
            '독립적이지만 가족과 함께 시간을 보내는 것도 좋아하는 고양이로,\n 순수하면서도 사자같은 외모인 당신이 부럽소!',
            '조용하고 느긋하게 집안에서 쉬는 것을 좋아하는 고양이로,\n 귀여운 외모를 가진 당신이 너무 부럽소!',
            '사람에게 잘 따르며 안기거나 옆에 누워 있는 것을 좋아하는 고양이로,\n 도도하면서도 귀여운 것이 당신과 딱 알맞소!',
            '조용하고 침착하게 지내며 관찰하는 것을 좋아하는 고양이로,\n 영리하게 생긴 당신을 보니 세상 참 불공평하오...',
            '놀이를 즐기며 가족과 함께 시간을 보내는 것을 좋아하는 고양이로,\n 순둥순둥하게 생긴게 당신과 찰떡일수도?!',
            '활발하고 수다스러우며 사람들과 상호작용하는 것을 좋아하는 고양이로,\n 외향적인 성향을 가진 당신과 비슷하오!',
            '활발하게 움직이며 따뜻한 곳에서 놀고 쉬는 것을 좋아하는 고양이로,\n 지적인 미모를 가졌소!',
            '호기심 많고 활발하게 집안을 돌아다니며 놀이를 즐기는 고양이로,\n 깨끗하면서 도도한 당신의 외모가 부럽소!']
    
    man_info = ['당신은 무쌍에 큰 눈, 희고 밝은 피부 톤을 가진 외모를 가졌소!\n겸손하고 현실적이고 가장 깔끔하고 말을 예쁘게 하는 사람이오.',
                '당신은 맑고 큰 눈과 웃을 때 나타나는 하트 모양의 입술을 가진 외모를 가졌소!\n말수가 적고 조용한 편이며 보통 다른 사람들의 말을 묵묵히 들어 주는 사람이오.',
                '당신은 동글동글하고 귀여운 느낌을 가진 외모를 가졌소!\n무관심해 보여도 뒤에서 다른 사람들을 챙겨주는 사려깊은 사람이오.',
                '오똑한 콧대가 만들어낸 T존, 선이 각지고 긴 외모를 가졌소!\n예술성이 뛰어나고 감정 기복이 크고 자기 스스로에게 만족하기 위해 노력하는 사람이오.']

    girl_info = ['당신은 작고 예쁜 두상과 매끄럽고 여백이 없는 외모를 가졌소!\n차갑고 도도해 보이지만, 귀여운 성격을 보유하고 있는 사람이오.',
                 '순한 눈매와 온순한 인상이 특징인 외모를 가졌소!\n애교가 많은 타입은 아니지만 사람들 사이에서 웃음을 자아내는 사람이오.',
                 '얇은 턱선과 얼굴형, 장두형에 가까운 두상을 가진 외모를 가졌소!\n시크한 냉미녀상 외모이지만, 이미지와는 정반대로 다정다감하고 세심한 사람이오.',    
                 '모든 컨셉을 자유롭게 소화 가능한 올라운더 페이스를 가진 외모를 가졌소!\n전반적으로 밝고 텐션이 높아 본인이 나서서 분위기를 이끌어 가지만 낯을 제법 가리는 편이오.']


    dog_img = ['img/dog/beagle.jpg','img/dog/chiwawa.jpg','img/dog/golden.jpg',
               'img/dog/pome.jpg','img/dog/pug.jpeg','img/dog/huski.jpg']
    
    cat_img = ['img/cat/Abyssinian.jpg','img/cat/American Bobtail.jpg','img/cat/American Curl.jpg',
               'img/cat/American Shorthair.jpg','img/cat/Bengal.jpg','img/cat/Birman.jpg',
               'img/cat/Bombay.jpg','img/cat/British Shorthair.jpg','img/cat/Egyptian Mau.jpg',
               'img/cat/Exotic Shorthair.jpg','img/cat/Maine Coon.jpg','img/cat/Manx.jpg',
               'img/cat/Norwegian Forest.jpg','img/cat/Persian.jpg','img/cat/Ragdoll.jpg',
               'img/cat/Russian Blue.jpg','img/cat/Scottish Fold.jpg','img/cat/Siamese.jpg',
               'img/cat/Sphynx.jpg','img/cat/Turkish Angora.jpg']
    
    man_img = ['img/man/시우민.jpeg','img/man/디오.jpeg','img/man/슈가.jpeg','img/man/탑.jpeg']
    
    girl_img = ['img/girl/제니.jpeg','img/girl/카리나.jpeg','img/girl/지수.jpeg','img/girl/윈터.jpeg']
    
    ['비글','치와와','골든 리트리버','포메라니안','퍼그','시베리안 허스키']


    if dog_pred:
        dog_idx = int(request.session.get('dog_idx'))
        return render(request, 'face/result_page.html', {'pred1': dog_pred, 'face':dog_face[dog_idx],
                                                         'pers':dog_personality[dog_idx], 'info1':dog_info[dog_idx][:dog_info[dog_idx].index('\n')],
                                                         'info2':dog_info[dog_idx][dog_info[dog_idx].index('\n'):],
                                                         'r_image':dog_img[dog_idx]})
    elif cat_pred:
        cat_idx = int(request.session.get('cat_idx'))
        return render(request, 'face/result_page.html', {'pred2': cat_pred, 'face':cat_face[cat_idx],
                                                         'pers':cat_personality[cat_idx], 'info1':cat_info[cat_idx][:cat_info[cat_idx].index('\n')],
                                                         'info2':cat_info[cat_idx][cat_info[cat_idx].index('\n'):],
                                                         'r_image':cat_img[cat_idx]})

    elif man_pred:
        man_idx = int(request.session.get('idx_man'))
        return render(request, 'face/result_page.html', {'pred3': man_pred,
                                                         'info1':man_info[man_idx][:man_info[man_idx].index('\n')],
                                                         'info2':man_info[man_idx][man_info[man_idx].index('\n'):],
                                                         'r_image':man_img[man_idx]})

    elif girl_pred:
        girl_idx = int(request.session.get('idx_girl'))
        return render(request, 'face/result_page.html', {'pred4': girl_pred,
                                                         'info1':girl_info[girl_idx][:girl_info[girl_idx].index('\n')],
                                                         'info2':girl_info[girl_idx][girl_info[girl_idx].index('\n'):],
                                                         'r_image':girl_img[girl_idx]})
