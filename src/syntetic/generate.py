import os
import re
import time
import random
import asyncio
import pandas as pd
import logging
from typing import List, Dict, Optional
from mistralai import Mistral

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    def __init__(self):
        # Initialize Mistral client
        self.api_key = "Mw1Qh5yjUhi3Ko5T3YQoegIKCpZDATar" #os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable must be set")
        
        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-large-latest"  # Can be changed to other models as needed
        
        # Generation parameters
        self.api_delay = 5  # Delay between API calls in seconds
        self.max_attempts = 3  # Maximum attempts per technique
        
        # Define manipulation techniques and their descriptions
        self.technique_descriptions = {
            'loaded_language': "The use of words and phrases with a strong emotional connotation (positive or negative) to influence the audience.",
            'glittering_generalities': 'Exploitation of people\'s positive attitude towards abstract concepts such as "justice," "freedom," "democracy," "patriotism," "peace," etc., intended to provoke strong emotional reactions without specific information.',
            'euphoria': "Using an event that causes euphoria or a feeling of happiness, or a positive event to boost morale, often used to mobilize the population.",
            'appeal_to_fear': "The misuse of fear (often based on stereotypes or prejudices) to support a particular proposal.",
            'fud': "Presenting information in a way that sows uncertainty and doubt, causing fear. A subtype of appeal to fear.",
            'bandwagon': 'An attempt to persuade the audience to join and take action because "others are doing the same thing."',
            'cliche': 'Commonly used phrases that mitigate cognitive dissonance and block critical thinking (e.g., "Time will tell," "It\'s not all black and white").',
            'whataboutism': "Discrediting the opponent's position by accusing them of hypocrisy without directly refuting their arguments.",
            'cherry_picking': "Selective use of data or facts that support a hypothesis while ignoring counterarguments.",
            'straw_man': "Distorting the opponent's position by replacing it with a weaker or outwardly similar one and refuting it instead."
        }
        
        # Examples for some techniques
        self.technique_examples = {
            'loaded_language': 'Example: "Военкомы продолжают паковать украинцев в микроавтобусы, пока нардепы только хотят рассмотреть нарушения в мобилизационном треке. Чем дольше власть тянет с решением этой проблемы, тем больше в обществе напоминают негативные тенденции."',
            'glittering_generalities': 'Example: "Дорогі українці! Сьогодні ми всі – єдині, як ніколи. У нас – один біль. За нашу державу. І одна мета – звільнити Україну від навали орків. Відстояти своє майбутнє. Київ серцем з мешканцями Маріуполя, Миколаєва, Харкова, Херсона, Чернігова, Житомира, інших міст і містечок України. Разом ми сильні! Наші військові – відважні герої! Українці – вільна нація. І ми вистоїмо! Слава Україні!"',
            'euphoria': 'Example: "«Поймали на оперативном скачке», – Арестович об успешном контрнаступлении украинской армии на Херсонском направлении. Наши влупили на 3 направлениях, 3 переправы организовали – все успешные. Форсировали Ингулец, закрепились, создали плацдарм. Разгромили батальонно-тактическую мотострелковую группу врага в ноль."',
            'appeal_to_fear': 'Example: "УКРАЇНЦІВ ЗМУСЯТЬ СТАТИ НА ОБЛІК ЗА КОРДОНОМ? Українців за кордоном, які не стали на облік, хочуть позбавляти банківських та консульських послуг, — депутат Вадим Івченко. «Банк запросить також нову ідентифікацію, але в цій ідентифікації має бути це посвідчення, безпосередньо цей військовий облік. Це означає, що можуть вимкнутись навіть банківські картки.»"',
            'fud': 'Example: "Останню бригаду кинули в бій, резервів не залишилось…" – підваження віри українців у власну державу чи перемогу; страх серед сил оборони."',
            'bandwagon': 'Example: "В Германии набирает тренд санкционный скептицизм, чем больше вводят ограничений для немцев тем больше они настроены против санкционной политики. Уже в открытую звучат призывы задействовать Северный поток-2 для обеспечения газом Германию…"',
            'cliche': 'Example: "Що тут коментувати? У нас зараз завдання одне – перемогти. Ми воюємо. Що там хто пише, піднімає той бруд з колін – то його проблеми. Все це – елемент інформаційної війни. Час все розставить на свої місця. Я публічна людина, витримаю все. Весь цей процес – нісенітниця"',
            'whataboutism': 'Example: "Херсонская область, Скадовск (территория подконтрольна РФ). ВСУ выпустили точку-У, россияне её сбили, она упала на жилой дом, погибли мирные люди. Офис Президента через ботофермы запустил контртезис, что ставить ПВО в городе - это преступление россиян. Такая же ситуация была в Одессе, когда только уже РФ запустили ракеты, ПВО ВСУ сбило её над городом, тогда погиб младенец и девушка. Исходя логики ботов ОП - это тоже преступление, только ВСУ? Или это другое…"',
            'cherry_picking': 'Example: "По нашим данным военкомы Одесской, Полтавской, Кировоградской, Днепропетровской, Харьковской, Черкасской областей, почти перевыполняют план по мобилизации. Усилено хватают всех. К примеру западная Украина срывает все мобилизационные процессы, не выполняя план даже на 50%."',
            'straw_man': 'Example: "Сьогодні монобільшість очікувано ухвалила закон про марихуану. Ще одне рішення проти майбутнього України. Ми протистояли цій авантюрі усіма засобами, але змогли лише відтермінувати неминуче – занадто вже нерівними були сили. Та тим, хто вже ділить нашу землю на наркоплантації, рано тріумфувати. Вірю, що вже наступне скликання Верховної Ради скасує не лише легітимізацію наркотиків, а й всі інші закони проти національних інтересів та засад існування України."'
        }
        
    def create_prompt_messages(self, technique_name: str, num_examples: int) -> List[Dict]:
        """Create a detailed prompt as a list of messages for the Mistral API."""
        description = self.technique_descriptions.get(technique_name, "A specific manipulation technique.")
        example = self.technique_examples.get(technique_name, "No example provided.")

        system_prompt = """You are an expert specialized in generating realistic examples of text manipulation techniques, specifically within the context of Ukrainian news regarding ot the ongoing war in Ukraine, politic and society, mobilization. Your goal is to create training data for Ukrainian language."""

        user_prompt = f"""Generate exactly {num_examples} diverse and realistic examples of text content that clearly demonstrate the '{technique_name}' manipulation technique.

Definition of '{technique_name}': {description}
Example of '{technique_name}': {example}

Generation Instructions:
1.  Provide exactly {num_examples} distinct examples.
2.  Each example should be around 200 world, like a social media post, news snippet, or quote.
3.  The examples MUST clearly use the '{technique_name}' technique.
4.  Content must be relevant to the Ukrainian context (politics, society, war, international relations). Write in Unrainian reflecting propaganda are acceptable if appropriate for the technique.
5.  Ensure diversity in topic and phrasing across examples.
6.  Do NOT include explanations within the generated text itself.
7.  Format the output STRICTLY as a numbered list, starting immediately (no introductory sentences).

Example Output Format:
1. [Example text 1 using the technique]
2. [Example text 2 using the technique]
...
{num_examples}. [Example text {num_examples} using the technique]

Generate the examples now:
"""
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
    
    async def generate_examples(self, technique: str, num_examples: int) -> List[Dict]:
        """Generate examples for a specific technique."""
        logger.info(f"Generating data for: {technique}...")
        messages = self.create_prompt_messages(technique, num_examples)
        attempts = 0
        generated_texts = []
        synthetic_data = []

        while attempts < self.max_attempts and len(generated_texts) < num_examples:
            try:
                # Use asyncio to handle async calls
                chat_response = await asyncio.to_thread(
                    self.client.chat.complete,
                    model=self.model,
                    messages=messages,
                    temperature=1
                )

                # Extract text content from the response
                raw_text = chat_response.choices[0].message.content

                # Parse the results assuming numbered list format
                potential_examples = raw_text.strip().split('\n')
                current_examples = []
                for line in potential_examples:
                    # Remove number prefix and strip whitespace
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip()).strip()
                    if cleaned_line:  # Avoid empty lines
                        current_examples.append(cleaned_line)

                logger.info(f"  Attempt {attempts+1}: Received {len(current_examples)} potential examples from API.")

                # Add generated examples to list for this technique
                needed = num_examples - len(generated_texts)
                generated_texts.extend(current_examples[:needed])

                if len(generated_texts) >= num_examples:
                    logger.info(f"  Successfully generated {len(generated_texts)} examples for {technique}.")
                    break  # Got enough examples

                attempts += 1
                if attempts < self.max_attempts:
                    logger.info(f"  Generated {len(generated_texts)}/{num_examples}. Retrying for more...")
                    await asyncio.sleep(self.api_delay + random.uniform(0, 2))  # Add jitter to delay

            except Exception as e:
                attempts += 1
                logger.error(f"  ERROR during Mistral API call or parsing (Attempt {attempts}/{self.max_attempts}): {e}")
                if attempts >= self.max_attempts:
                    logger.warning(f"  Max attempts reached for {technique}. Moving to next technique.")
                else:
                    # Consider longer delay or backoff on error
                    await asyncio.sleep(self.api_delay * (attempts + 1))

        # Create the data entries for successful examples
        for text in generated_texts:
            synthetic_data.append({'content': text, 'techniques': [technique]})

        return synthetic_data
    
    async def generate_dataset(self, techniques: List[str], num_examples_per_technique: int) -> pd.DataFrame:
        """Generate synthetic dataset for the specified techniques."""
        logger.info(f"Starting synthetic data generation using Mistral model: {self.model}")
        logger.info(f"Techniques to augment: {techniques}")
        logger.info(f"Examples per technique: {num_examples_per_technique}")
        
        all_synthetic_data = []
        
        for technique in techniques:
            # Generate examples for this technique
            technique_data = await self.generate_examples(technique, num_examples_per_technique)
            all_synthetic_data.extend(technique_data)
            
            # Wait before processing the next technique
            logger.info(f"  Waiting {self.api_delay} seconds before next technique...")
            await asyncio.sleep(self.api_delay)
        
        # Create DataFrame from generated data
        if all_synthetic_data:
            df_synthetic = pd.DataFrame(all_synthetic_data)
            # Ensure 'techniques' column is object type to hold lists
            df_synthetic['techniques'] = df_synthetic['techniques'].astype('object')
        else:
            logger.warning("No synthetic data was generated.")
            # Create empty dataframe with correct columns
            df_synthetic = pd.DataFrame(columns=['content', 'techniques'])
        
        # Display summary statistics
        logger.info(f"Total synthetic examples generated: {len(df_synthetic)}")
        if not df_synthetic.empty:
            logger.info("Synthetic Data Counts per Technique:")
            technique_counts = df_synthetic['techniques'].apply(lambda x: x[0]).value_counts()
            for tech, count in technique_counts.items():
                logger.info(f"  {tech}: {count}")
        
        return df_synthetic
    
    async def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """Save the generated dataset to a CSV file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Synthetic dataset saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")

async def main():
    # Techniques to focus on
    techniques_to_augment = [
        'straw_man',
        'appeal_to_fear',
        'fud',
        'bandwagon',
        'whataboutism',
        'glittering_generalities',
        'euphoria',
        'cherry_picking',
        'cliche'
    ]
    
    # Number of examples per technique
    num_examples_per_technique = 25
    
    # Output file path
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'synthetic_data.csv')
    
    try:
        # Initialize the generator
        generator = SyntheticDataGenerator()
        
        # Generate the dataset
        df_synthetic = await generator.generate_dataset(techniques_to_augment, num_examples_per_technique)
        
        # Save the dataset
        await generator.save_to_csv(df_synthetic, output_path)
        
    except Exception as e:
        logger.error(f"Error in synthetic data generation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
