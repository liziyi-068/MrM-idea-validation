
import os
import base64
from PIL import Image, ImageDraw
from zhipuai import ZhipuAI


ZHIPU_API_KEY = "####"

client = ZhipuAI(api_key=ZHIPU_API_KEY)
MODEL_NAME = "glm-4v-plus"  


OD_NAME_PROMPT = (
    "Analyze the precise positional correspondence between the red-masked region in the upper image and the reference image below. "
    "Extract ONLY the visual content from the EXACT SAME POSITION in the lower reference image. "
    "Output ONLY a word or phrase. DO NOT WRITE ANYTHING ELSE."
)

def apply_mask(image_path, output_path, mask_box, mask_color='red'):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    x, y, w, h = mask_box
    fill_color = (255, 0, 0) if mask_color == 'red' else (0, 0, 0)
    draw.rectangle([(x, y), (x + w, y + h)], fill=fill_color)
    img.save(output_path)
    return output_path

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def query_model(image_path, prompt):
    base64_img = encode_image(image_path)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        }],
        temperature=0,
        max_tokens=20
    )
    return response.choices[0].message.content.strip().lower()

def mrm_attack(image_path, true_label, mask_box, ref_image_path, trials=3):
    masked_path = "temp_masked.jpg"
    apply_mask(image_path, masked_path, mask_box, mask_color='red')
    
    img1 = Image.open(masked_path)
    img2 = Image.open(ref_image_path)
    total_width = max(img1.width, img2.width)
    total_height = img1.height + img2.height + 10
    combined = Image.new('RGB', (total_width, total_height), (0, 0, 0))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (0, img1.height + 10))
    combined_path = "temp_combined.jpg"
    combined.save(combined_path)
    
    answers = []
    for i in range(trials):
        ans = query_model(combined_path, OD_NAME_PROMPT)
        answers.append(ans)
        print(f"  尝试 {i+1}: {ans}")
    
    correct = sum(1 for ans in answers if true_label.lower() in ans)
    accuracy = correct / trials
    
    for f in [masked_path, combined_path]:
        if os.path.exists(f):
            os.remove(f)
    
    return accuracy, answers

if __name__ == "__main__":
    member_img = "member.jpg"
    nonmember_img = "nonmember.jpg"
    ref_img = "reference.jpg"
    mask_box = (100, 100, 150, 150)
    
    print("测试成员图片...")
    acc_mem, _ = mrm_attack(member_img, "leaves", mask_box, ref_img, trials=3)  
    print(f"成员图片准确率: {acc_mem:.2f}")
    
    print("\n测试非成员图片...")
    acc_non, _ = mrm_attack(nonmember_img, "leaves", mask_box, ref_img, trials=3) 
    print(f"非成员图片准确率: {acc_non:.2f}")
    
    print("\n结论:")
    if acc_mem > acc_non:
        print("✅ 成员图片答对率更高，MrM 核心逻辑成立")
    else:
        print("⚠️ 差异不明显")