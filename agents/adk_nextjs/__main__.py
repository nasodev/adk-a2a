import logging
import os

import click

from agent import NextjsAgent
from common.server import A2AServer
from common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MissingAPIKeyError,
)
from dotenv import load_dotenv
from task_manager import AgentTaskManager


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=13002)
def main(host, port):
    try:
        # Check for API key only if Vertex AI is not configured
        if not os.getenv('GOOGLE_GENAI_USE_VERTEXAI') == 'TRUE':
            if not os.getenv('GOOGLE_API_KEY'):
                raise MissingAPIKeyError(
                    'GOOGLE_API_KEY environment variable not set and GOOGLE_GENAI_USE_VERTEXAI is not TRUE.'
                )

        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id='nextjs_development',
            name='NextJS Development Assistant',
            description='Helps with NextJS development including project setup, component creation, routing, data fetching, and best practices implementation.',
            tags=['nextjs', 'react', 'frontend', 'web'],
            examples=[
                'How do I set up a new NextJS project with TypeScript?',
                'Help me create a responsive navigation component in NextJS',
                'What is the best way to implement API routes in NextJS?',
                'Create a data fetching component using SWR in NextJS'
            ],
        )
        agent_card = AgentCard(
            name='NextJS Development Agent',
            description='This agent assists with NextJS development, providing guidance on project setup, component design, routing configuration, state management, and frontend best practices.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=NextjsAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=NextjsAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=NextjsAgent()),
            host=host,
            port=port,
        )
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()
