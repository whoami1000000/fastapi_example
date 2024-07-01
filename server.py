import asyncio
import logging
import random
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from time import sleep

import starlette.status
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s][%(levelname)s]:\t%(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

NEW = 'new'
RUNNING = 'running'
DONE = 'done'
FAILED = 'failed'

statuses: dict[str, str] = {}
results: dict[str, float] = {}


def init_ml_model():
    # init everything here
    # read some data, etc
    pass


@asynccontextmanager
async def init(_: FastAPI):
    init_ml_model()

    app.state.executor = ProcessPoolExecutor()

    yield

    app.state.executor.shutdown()


app = FastAPI(lifespan=init, debug=True)


class SimulationRequest(BaseModel):
    class StepData(BaseModel):
        step: int
        currents: list[float]

    alpha_p: float
    alpha_ff: float
    beta_p: float
    beta_ff: float
    r_mag: float
    z_mag: float
    center: float
    ip: float
    b_tor: float
    steps_num: int
    steps: list[StepData]


class StatusResponse(BaseModel):
    id: str
    status: str


def do_real_job(simulation_id: str, data: SimulationRequest) -> float:
    sleep(15)  # like blocking func
    return random.random()


async def run_in_process(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(app.state.executor, fn, *args)


async def run_simulation(simulation_id: str, data: SimulationRequest):
    logger.info(f'start simulation [id={simulation_id}]')

    statuses[simulation_id] = RUNNING

    result = await run_in_process(do_real_job, simulation_id, data)

    logger.info(f'finish simulation [id={simulation_id}][res={result}]')

    results[simulation_id] = result
    statuses[simulation_id] = DONE  # or FAILED


@app.post('/simulate')
async def simulate(_: Request, data: SimulationRequest, background_tasks: BackgroundTasks):
    simulation_id = str(uuid.uuid4())

    statuses[simulation_id] = NEW

    background_tasks.add_task(run_simulation, simulation_id, data)

    return JSONResponse(status_code=starlette.status.HTTP_201_CREATED,
                        content=StatusResponse(id=simulation_id, status=NEW).dict())


@app.get('/status')
async def get_status(request: Request):
    simulation_id = request.query_params['id']
    status = statuses[simulation_id]
    return JSONResponse(status_code=starlette.status.HTTP_200_OK,
                        content=StatusResponse(id=simulation_id, status=status).dict())


if __name__ == '__main__':
    uvicorn.run(app, port=8001)
