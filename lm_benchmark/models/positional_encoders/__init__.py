# Copyright 2023 Amirkeivan Mohtashami, Martin Jaggi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import encoder, rotary, rotary_mem_jump

POS_ENCS = {
    "rotary": rotary.RotaryPositionalEncoder,
    "rotary_mem_jump": rotary_mem_jump.RotaryJumpMemPositionalEncoder
}


def get_encoder(encoder_name):
    return POS_ENCS[encoder_name]


def registered_encoders():
    return POS_ENCS.keys()
